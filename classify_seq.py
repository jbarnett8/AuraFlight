import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence, PackedSequence
import matplotlib.pyplot as plt
from tqdm import tqdm
from classify_labelled import PackedMaskedLayerNorm, TransformerClassifier, TimeSeriesClassifier

# Set up matplotlib to use LaTeX
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

def get_args():
	parser = argparse.ArgumentParser('Classify Aircraft Maneuvers in Sequence')
	parser.add_argument('--input_file', type=str, required=True,
	                    help='Path to the input numpy file (.npy)')
	parser.add_argument('--column_names', type=str, required=True,
	                    help='Path to a text file containing column names, one per line')
	parser.add_argument('--model_path', type=str, required=True,
	                    help='Path to the trained model checkpoint (.pth)')
	parser.add_argument('--selected_columns', nargs='+', default=None,
	                    help='List of column names to use for classification')
	parser.add_argument('--lat_col', type=str, default='Latitude',
	                    help='Name of latitude column for plotting')
	parser.add_argument('--lon_col', type=str, default='Longitude',
	                    help='Name of longitude column for plotting')
	parser.add_argument('--alt_col', type=str, default='Altitude',
	                    help='Name of altitude column for plotting')
	parser.add_argument('--confidence_threshold', type=float, default=0.8,
	                    help='Minimum confidence for predictions')
	parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'lstm'],
	                    help='Type of model (transformer or lstm)')
	parser.add_argument('--num_heads', type=int, default=1,
	                    help='Number of attention heads for transformer')
	parser.add_argument('--dropout', type=float, default=0.0,
	                    help='Dropout probability')
	parser.add_argument('--num_lstm_layers', type=int, default=2,
	                    help='Number of LSTM layers')
	parser.add_argument('--num_classes', type=int, default=4,
	                    help='Number of classes in the model')
	parser.add_argument('--lstm_n', type=int, default=10,
	                    help='Number of LSTM layer units')
	parser.add_argument('--dense_n', type=int, default=10,
	                    help='Number of dense layer units')
	parser.add_argument('--label_names', nargs='+', default=['Chandelle', 'Landing', 'Steep Turn', 'Takeoff'],
	                    help='Names of the classes in order')
	parser.add_argument('--output_prefix', type=str, default='prediction',
	                    help='Prefix for output file names')

	args = parser.parse_args()
	return args


def load_data(input_file, column_names_file, selected_columns):
	# Load the data
	data = np.load(input_file)[40:, :] # Skip the first 20 rows since good data is after that
	reindex = [0, 3, 2, 1, 4, 5]
	data = data[:, np.array(reindex)]

	# Unwrap the heading
	data[:, 3] = np.unwrap(data[:, 3], period=360)

	# Load column names
	with open(column_names_file, 'r') as f:
		all_columns = [line.strip() for line in f.readlines()]
		all_columns = [all_columns[v] for v in reindex]

	time = data[:, all_columns.index('Time')]

	data = np.array([np.interp(np.arange(np.min(time), np.max(time)), time, v) for v in data.T]).T
	time = data[:, all_columns.index('Time')]

	# Get indices for latitude, longitude, altitude for plotting
	lat_idx = all_columns.index('Latitude') if 'Latitude' in all_columns else None
	lon_idx = all_columns.index('Longitude') if 'Longitude' in all_columns else None

	# Create plotting data
	plot_data = None
	if lat_idx is not None and lon_idx is not None:
		plot_data = np.column_stack((time, data[:, lat_idx], data[:, lon_idx], data[:, all_columns.index('Altitude')]))

	# Select columns if specified
	if selected_columns:
		# Find column indices
		column_indices = [all_columns.index(col) for col in selected_columns if col in all_columns]
		if len(column_indices) != len(selected_columns):
			missing = [col for col in selected_columns if col not in all_columns]
			print(f"Warning: Some requested columns were not found: {missing}")
		data = data[:, column_indices]

	alt_idx = all_columns.index('Altitude') - 1 if 'Altitude' in all_columns else None
	reindex = np.array([2, 1, 0])
	return data[:, reindex], plot_data, alt_idx


def classify_sequence(model, data, plot_data, alt_data, args):
	device = torch.device(
		'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
	model.to(device)
	model.eval()

	# Convert to tensor
	data_tensor = torch.FloatTensor(data)

	# Parameters for sliding window prediction
	max_seq_length = 30
	step = 10

	# Minimum length for a maneuver to be valid
	min_maneuver_length = 20

	# Make predictions for each timestep
	predictions = []
	confidences = []  # Store confidence values
	print(f"Processing sequence, length: {len(data_tensor)}")

	with torch.no_grad():
		for t in tqdm(range(step, len(data_tensor) + step, step)):  # Process every nth timestep
			# Get sequence up to current timestep
			if t > len(data_tensor):
				t = len(data_tensor)

			start = max(0, t - max_seq_length)
			seq_up_to_t = data_tensor[start:t].clone()

			if len(seq_up_to_t) == 0:
				continue

			# Pack the sequence for the model
			packed_seq = pack_sequence([seq_up_to_t], enforce_sorted=False).to(device)

			# Get prediction
			output = model(packed_seq)

			# Get predicted class and confidence
			softmax_output = F.softmax(output, dim=-1)
			confidence, predicted = torch.max(softmax_output, 1)

			# Store predictions and confidences
			timestep_count = t % step if t % step != 0 else step
			predicted_like_seq = np.ones(timestep_count) * predicted.item()
			confidence_like_seq = np.ones(timestep_count) * confidence.item()

			predictions.append(predicted_like_seq)
			confidences.append(confidence_like_seq)

	# Convert predictions and confidences to tensors
	predictions_tensor = np.hstack(predictions)
	confidences_tensor = np.hstack(confidences)

	# Create confidence mask for filtering
	confidence_mask = confidences_tensor >= args.confidence_threshold

	# Create smoothed predictions by filling gaps between same maneuver types
	smoothed_predictions = predictions_tensor.copy()
	smoothed_mask = confidence_mask.copy()

	# For each label type, find and fill gaps between consecutive occurrences
	for label_idx in range(args.num_classes):
		# Find all positions where this label appears with high confidence
		high_conf_label_positions = np.where((smoothed_predictions == label_idx) & smoothed_mask)[0]

		if len(high_conf_label_positions) < 2:
			continue  # Need at least two occurrences to have a gap

		# Find gaps between consecutive occurrences
		gaps = np.where(np.diff(high_conf_label_positions) > 1)[0]

		for gap_idx in gaps:
			start_pos = high_conf_label_positions[gap_idx] + 1
			end_pos = high_conf_label_positions[gap_idx + 1] - 1

			# Check if gap is not too large
			max_gap_size = 30  # Maximum timesteps to fill
			if end_pos - start_pos > max_gap_size:
				continue

			# Check if the gap only contains predictions below confidence threshold
			if not np.any(smoothed_mask[start_pos:end_pos + 1]):
				smoothed_predictions[start_pos:end_pos + 1] = label_idx
				smoothed_mask[start_pos:end_pos + 1] = True

	# Now identify and discard short segments
	for label_idx in range(args.num_classes):
		# Get binary mask for this label
		label_mask = smoothed_predictions == label_idx

		# Find continuous segments
		# 1 where the value changes, 0 elsewhere
		change_points = np.diff(np.concatenate(([0], label_mask.astype(int), [0])))
		# Indices where segments start
		starts = np.where(change_points == 1)[0]
		# Indices where segments end
		ends = np.where(change_points == -1)[0]

		# Check each segment's length
		for segment_idx in range(len(starts)):
			start_pos = starts[segment_idx]
			end_pos = ends[segment_idx]
			segment_length = end_pos - start_pos

			# If segment is too short, discard it by setting the mask to False
			if segment_length < min_maneuver_length:
				smoothed_mask[start_pos:end_pos] = False

	# Create plots
	create_plots(plot_data, alt_data, predictions_tensor, confidences_tensor,
	             smoothed_predictions, smoothed_mask, args)


def create_plots(plot_data, alt_data, predictions, confidences,
                 smoothed_predictions, smoothed_mask, args):
	"""
	Create plots for visualization of maneuver classification
	"""
	# Get label names
	label_names = args.label_names

	# If we have latitude/longitude data, create lat/lon plots
	if plot_data is not None:
		fig, ax = plt.subplots(figsize=(8, 3))

		# Plot full path as light gray line for context
		ax.plot(
			plot_data[:len(smoothed_predictions), 2],
			plot_data[:len(smoothed_predictions), 1],
			'-', color='lightgray', alpha=0.5, linewidth=0.7, label='Trajectory'
		)

		# Plot predictions with smoothed mask
		colors = plt.get_cmap('tab10', args.num_classes)
		markers = ['o', 's', 'D', '^', 'v', 'x']
		for label_idx in range(args.num_classes):
			mask = (smoothed_predictions == label_idx) & smoothed_mask
			if np.any(mask):
				label_name = label_names[label_idx] if label_idx < len(label_names) else f"Class {label_idx}"
				ax.plot(
					plot_data[mask, 2],
					plot_data[mask, 1],
					markers[label_idx], label=label_name, color=colors(label_idx), 
					markersize=1, markevery=0.02
				)

		ax.set_xlabel(r'Longitude($^\circ$)')
		ax.set_ylabel(r'Latitutde ($^\circ$)')
		ax.set_aspect('equal')
		ax.legend(fontsize=7)
		ax.grid(True, linestyle='--', alpha=0.7)

		plt.tight_layout()
		plt.savefig(f"{args.output_prefix}_lat_lon_predictions.png", dpi=300)
		plt.close()

	# If we have altitude data, create altitude vs time plot with a broken y-axis
	if alt_data is not None:
		# Create figure with a special GridSpec layout for the broken axis
		fig = plt.figure(figsize=(8, 2.5))
		gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.1)
		ax1 = fig.add_subplot(gs[0])  # Top subplot for high altitudes
		ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Bottom subplot for low/mid altitudes

		# Create time values (x-axis)
		time_values = plot_data[:len(smoothed_predictions), 0]
		altitude_data = plot_data[:len(smoothed_predictions), 3]

		# Determine altitude ranges for break point
		alt_min = np.min(altitude_data)
		alt_max = np.max(altitude_data)
		break_point = alt_min + (alt_max - alt_min) * 0.80  # Adjust this to set where the break occurs

		# Define y-limits for each section
		top_range = (break_point + 100, alt_max + 100)  # Add a small buffer
		bottom_range = (alt_min - 100, break_point - 100)  # Add a small buffer

		# Define maneuver regions as (start_time, end_time, maneuver_name)
		maneuver_regions = [
			(time_values[0], time_values[0] + 400, "Takeoff"),
			(650, 950, "Steep Turn"),
			(950, 1000, "Chandelle"),
			(1090, 1125, "Chandelle"),
			(1200, time_values[-1], "Landing")
		]

		# Define different hatch patterns for different maneuvers
		hatch_patterns = {
			"Takeoff": '///',
			"Landing": '\\\\\\',
			"Steep Turn": '...',
			"Chandelle": 'xx',
		}

		# Define colors with transparency for the regions
		region_colors = {
			"Takeoff": (0.2, 0.2, 0.8, 0.1),
			"Landing": (0.8, 0.2, 0.2, 0.1),
			"Steep Turn": (0.2, 0.8, 0.2, 0.1),
			"Chandelle": (0.8, 0.8, 0.2, 0.1),
		}

		# Before adding the maneuver regions, set hatch properties
		original_linewidth = plt.rcParams['hatch.linewidth']
		original_color = plt.rcParams['hatch.color']

		# Make the hatch lines thinner and semi-transparent
		plt.rcParams['hatch.linewidth'] = 0.5
		plt.rcParams['hatch.color'] = 'gray'

		# Add the maneuver regions to both subplots
		legend_elements = []
		for start_time, end_time, maneuver in maneuver_regions:
			rect1 = ax1.axvspan(start_time, end_time,
			                    facecolor=region_colors.get(maneuver, (0.5, 0.5, 0.5, 0.1)),
			                    hatch=hatch_patterns.get(maneuver, '//'),
			                    alpha=0.2, zorder=0)

			# Add rectangle to bottom subplot
			rect2 = ax2.axvspan(start_time, end_time,
			                    facecolor=region_colors.get(maneuver, (0.5, 0.5, 0.5, 0.1)),
			                    hatch=hatch_patterns.get(maneuver, '//'),
			                    alpha=0.2, zorder=0)

			if f'{maneuver} (Actual)' not in [item.get_label() for item in legend_elements]:
				legend_elements.append(plt.Rectangle((0, 0), 1, 1,
				                                     facecolor=region_colors.get(maneuver, (0.5, 0.5, 0.5, 0.1)),
				                                     hatch=hatch_patterns.get(maneuver, '//'),
				                                     alpha=0.2, label=f"{maneuver} (Actual)"))

		# Reset hatch properties back to original values
		plt.rcParams['hatch.linewidth'] = original_linewidth
		plt.rcParams['hatch.color'] = original_color

		for ax in [ax1, ax2]:
			ax.plot(
				time_values[:len(smoothed_predictions)],
				altitude_data,
				'-', color='lightgray', alpha=0.5, linewidth=0.7, label='Altitude Profile'
			)

			# Plot predicted labels with altitude
			colors = plt.get_cmap('tab10', args.num_classes)
			markers = ['o', 's', 'D', '^', 'v', 'x']
			for label_idx in range(args.num_classes):
				mask = (smoothed_predictions == label_idx) & smoothed_mask
				if np.any(mask):
					label_name = label_names[label_idx] if label_idx < len(label_names) else f"Class {label_idx}"
					ax.plot(
						time_values[mask],
						plot_data[mask, 3],
						markers[label_idx], label=label_name, color=colors(label_idx),
						markersize=2, markevery=0.01
					)

			ax.grid(True, linestyle='--', alpha=0.7)

		ax1.set_ylim(top_range)
		ax2.set_ylim(bottom_range)

		# Add break marks
		d = 0.015
		kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
		ax1.plot((-d, d), (-d, +d), **kwargs)  # Bottom-left diagonal
		ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Bottom-right diagonal

		kwargs.update(transform=ax2.transAxes)
		ax2.plot((-d, d), (1 - d, 1 + d), **kwargs)  # Top-left diagonal
		ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Top-right diagonal

		# Hide the spines between ax1 and ax2
		ax1.spines['bottom'].set_visible(False)
		ax2.spines['top'].set_visible(False)
		ax1.xaxis.tick_top()
		ax1.tick_params(labeltop=False)  # Don't put tick labels at the top

		# Add labels
		ax2.set_xlabel('Time (s)')
		fig.text(0.04, 0.5, 'Altitude (ft)', va='center', rotation='vertical')

		# Add legend (only once)
		handles, labels = ax1.get_legend_handles_labels()

		# Combine with maneuver region legend elements
		all_handles = handles + legend_elements
		all_labels = labels + [element.get_label() for element in legend_elements]

		fig.legend(all_handles, all_labels, fontsize=8, loc='upper right', bbox_to_anchor=(1., 0.85))

		plt.subplots_adjust(left=0.12, right=0.79, top=0.9, bottom=0.2)
		plt.subplots_adjust(hspace=0.05)  # Reduce space between subplots
		plt.savefig(f"{args.output_prefix}_altitude_time_predictions.png", dpi=300)
		plt.close()

	# Create additional visualization: confidence over time
	fig, ax = plt.subplots(figsize=(8, 4))

	time_values = np.arange(len(confidences))
	ax.plot(time_values, confidences, '-', color='blue', alpha=0.7, linewidth=0.8)
	ax.axhline(y=args.confidence_threshold, color='red', linestyle='--',
	           label=f'Confidence Threshold ({args.confidence_threshold})')

	ax.set_title('Model Prediction Confidence')
	ax.set_xlabel('Time Step (n)')
	ax.set_ylabel('Confidence Level')
	ax.set_ylim(0, 1)
	ax.legend(fontsize=7)
	ax.grid(True, linestyle='--', alpha=0.7)

	plt.tight_layout()
	plt.savefig(f"{args.output_prefix}_confidence.png", dpi=300)
	plt.close()


def main():
	args = get_args()

	# Load data
	feature_data, plot_data, alt_idx = load_data(args.input_file, args.column_names, args.selected_columns)

	# Get altitude data if available
	alt_data = None
	if alt_idx is not None:
		alt_data = feature_data[:, alt_idx]

	# Set up model arguments as a class for compatibility
	class ModelArgs:
		pass

	model_args = ModelArgs()
	model_args.dropout = args.dropout
	model_args.num_heads = args.num_heads
	model_args.num_lstm_layers = args.num_lstm_layers

	# Initialize model
	input_size = feature_data.shape[1]
	if args.model_type == 'transformer':
		model = TransformerClassifier(
			input_size=input_size,
			lstm_hidden_size=args.lstm_n,
			dense_hidden_size=args.dense_n,
			num_classes=args.num_classes,
			args=model_args
		)
	else:
		print('LSTM model type is not implemented in this script.')
		exit(1)

	# Load model weights
	model.load_state_dict(torch.load(args.model_path, map_location='cpu'))

	# Run classification
	classify_sequence(model, feature_data, plot_data, alt_data, args)

	print(f"Classification complete. Results saved with prefix: {args.output_prefix}")


if __name__ == '__main__':
	main()
