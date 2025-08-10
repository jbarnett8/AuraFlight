import numpy as np
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from data_loader import get_train_val_test
from torch.nn.utils.rnn import pack_sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence

import matplotlib.pyplot as plt
import seaborn as sns

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

# Define a colorblind-friendly palette
colorblind_palette = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
sns.set_palette(colorblind_palette)


class TransformerClassifier(nn.Module):
    """
    Transformer-based neural network for time series classification.

    This model uses a combination of transformer encoder layers with attention mechanisms
    to classify flight maneuver time series data.

    Args:
        input_size (int): Number of input features
        lstm_hidden_size (int): Size of hidden layers
        dense_hidden_size (int): Size of dense layers
        num_classes (int): Number of output classes
        args (argparse.Namespace): Additional arguments for model configuration
    """
    def __init__(self, input_size, lstm_hidden_size, dense_hidden_size, num_classes, args):
        super(TransformerClassifier, self).__init__()
        self.num_fourier_modes = 5
        self.ln2 = nn.LayerNorm(input_size)
        hidden_dim = input_size * 3
        self.p1 = nn.Linear(input_size, hidden_dim)
        self.p2 = nn.Linear(input_size, hidden_dim)
        self.transformer_1 = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=args.num_heads, dropout=args.dropout, batch_first=True)
        self.transformer_2 = nn.MultiheadAttention(embed_dim=lstm_hidden_size, num_heads=args.num_heads, dropout=args.dropout, batch_first=True)
        self.w_key = nn.Linear(in_features=hidden_dim, out_features=lstm_hidden_size)
        self.w_query = nn.Linear(in_features=hidden_dim, out_features=lstm_hidden_size)
        self.w_value = nn.Linear(in_features=hidden_dim, out_features=lstm_hidden_size)

        self.transformer_1_dense = nn.Sequential(nn.Linear(lstm_hidden_size, lstm_hidden_size), nn.ReLU(), nn.Linear(lstm_hidden_size, lstm_hidden_size), nn.LayerNorm(lstm_hidden_size))

        self.w_key_nor = nn.Linear(in_features=lstm_hidden_size, out_features=lstm_hidden_size)
        self.w_query_nor = nn.Linear(in_features=hidden_dim, out_features=lstm_hidden_size)
        self.w_value_nor = nn.Linear(in_features=lstm_hidden_size, out_features=lstm_hidden_size)

        self.dense = nn.Linear(lstm_hidden_size, dense_hidden_size)
        self.output = nn.Linear(dense_hidden_size, num_classes)
        self.dense2 = nn.Linear(lstm_hidden_size, dense_hidden_size)
        self.hidden = nn.Linear(dense_hidden_size, dense_hidden_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_nor, lens = pad_packed_sequence(x, batch_first=True)
        x = self.p2(x_nor - x_nor[:, 0, :].unsqueeze(1))  # Normalize by subtracting the first time step
        x_nor = self.p1(torch.diff(x_nor, dim=1, prepend=torch.zeros(x_nor.shape[0], 1, x_nor.shape[2], device=x_nor.device)))

        batch_size, seq_len = x.shape[0], x.shape[1]

        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)

        # Generate causal attention mask (lower triangular with zeros)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        causal_mask = causal_mask.to(x.device)

        # Apply transformer with padding mask
        transformer_out_f = self.transformer_1(
            query, key, value,
            attn_mask=causal_mask,
            need_weights=False
        )[0]

        transformer_out_f = self.transformer_1_dense(transformer_out_f)

        query_nor = self.w_query_nor(x_nor)
        key_nor = self.w_key_nor(transformer_out_f)
        value_nor = self.w_value_nor(transformer_out_f)

        transformer_out_f_nor = self.transformer_2(
            query_nor, key_nor, value_nor,
            attn_mask=causal_mask, need_weights=False)[0]

        # Extract the last non-padded token for each sequence
        transformer_out = transformer_out_f[range(len(lens)), lens - 1, :]
        transformer_out_nor = transformer_out_f_nor[range(len(lens)), lens - 1, :]

        dense_out = self.relu(self.dense(transformer_out) + self.relu(self.dense2(transformer_out_nor)))
        output = self.output(self.hidden(self.relu(dense_out)))
        return output


def get_args():
    parser = argparse.ArgumentParser('Run LSTM Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the CSV files')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lstm_n', type=int, default=10,
                        help='Number of LSTM layer units')
    parser.add_argument('--dense_n', type=int, default=10,
                        help='Number of dense layer units')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--selected_columns', nargs='+', default=None,
                        help='List of selected columns from the CSV files')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--transformer', action='store_true', help='Use transformer model instead of LSTM')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of attention heads for transformer')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--include_derivatives', action='store_true',
                        help='Include time derivatives of selected columns as additional features')
    parser.add_argument('--derivatives_only', action='store_true',
                        help='Use only the derivatives of selected columns as features')
    parser.add_argument('--augment_with_partials', action='store_true',
                        help='Augment training data with partial sequences')
    parser.add_argument('--partial_sequence_count', type=int, default=3,
                        help='Number of partial sequences to generate per full sequence during augmentation')
    args = parser.parse_args()
    return args


def train_model(model, train_loader, val_loader, num_epochs, args, patience=10, device=None):
    class_counts = torch.zeros(len(train_loader.dataset.dataset.label_dict))
    for _, labels in train_loader:
        for label in torch.argmax(labels, dim=1):
            class_counts[label] += 1
    
    # Convert counts to weights (less frequent classes get higher weights)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print("Class weights:", class_weights.tolist())
    
    # Use weighted cross entropy loss
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50)
    best_val_loss = float('inf')
    patience_counter = 0
    device = device or torch.device('cpu')

    acc_threshold_met = False

    model.to(device)
    epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.atleast_3d(labels)
            optimizer.zero_grad()
            outputs = model(inputs)  # Take the output of the last time step
            loss = criterion(outputs.unsqueeze(-1), torch.argmax(labels, dim=1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
            _, predicted = torch.max(F.softmax(outputs, dim=-1), 1)
            correct_train += (predicted == torch.argmax(labels, dim=1).squeeze()).sum().item()
            total_train += len(labels)

        train_loss /= len(train_loader.dataset)
        train_accuracy = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = torch.atleast_3d(labels)
                outputs = model(inputs)
                loss = criterion(outputs.unsqueeze(-1), torch.argmax(labels, dim=1))
                val_loss += loss.item() * len(inputs)
                _, predicted = torch.max(F.softmax(outputs, dim=-1), 1)
                correct_val += (predicted == torch.argmax(labels, dim=1).squeeze()).sum().item()
                total_val += len(labels)

        val_loss /= len(val_loader.dataset)
        val_accuracy = correct_val / total_val

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping')
            break

        if train_accuracy >= 0.95 and val_accuracy >= 0.95 and acc_threshold_met is False:
            print(f'Saving checkpoint due to high training and validation accuracy: {train_accuracy:.4f}, {val_accuracy:.4f}')
            torch.save(model.state_dict(), 'acc_threshold.pth')
            acc_threshold_met = True

        if epoch > 0 and epoch % 10 == 0:
            print(f'Saving checkpoint at epoch {epoch}')
            torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')

    return epoch, acc_threshold_met, criterion


def plot_confusion_matrix(y_true, y_pred, class_names):
    old_dict = {
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    }
    new_dict = {
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'axes.titlesize': 16,
        'axes.labelsize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 12,
        'figure.titlesize': 16
    }
    plt.rcParams.update(new_dict)
    cm = confusion_matrix(y_true, y_pred)
    import seaborn
    seaborn.heatmap(cm, fmt='d', annot=True, square=True,
                cmap='gray_r', vmin=0, vmax=0,  # set all to white
                linewidths=0.5, linecolor='k',  # draw black grid lines
                xticklabels=class_names, yticklabels=class_names,
                cbar=False)

    # re-enable outer spines
    sns.despine(left=False, right=False, top=False, bottom=False)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.rcParams.update(old_dict)


def test_timestep_classification(model, test_loader, model_name, device, confidence_threshold=0.8):
    if model_name[-3:] == 'pth':
        file_name = model_name
        save_name = model_name[:-3].split('/')[-1]
    else:
        file_name = model_name + '.pth'
        save_name = model_name.split('/')[-1]

    model.load_state_dict(torch.load(file_name))
    model.to(device)
    model.eval()

    # Get the original dataset
    dataset = test_loader.dataset.dataset
    reverse_label_dict = {v: k for k, v in dataset.label_dict.items()}

    # Process a few samples from the test set
    for i in range(min(3, len(dataset.full_sequences))):
        print(f'i = {i}')
        seq_id = i
        print(f'len of full_sequences: {len(dataset.full_sequences)}')
        print(f'\tlen of full sequence: {len(dataset.full_sequences[i])}')
        full_sequence = dataset.full_sequences[seq_id]
        true_labels = dataset.timestep_labels[seq_id]

        lat_long_data = dataset.full_sequences_lat_long[seq_id]

        selected_columns = dataset.selected_columns
        alt_idx = selected_columns.index('Altitude') if 'Altitude' in selected_columns else None

        predictions = []
        confidences = []
        print(f"Processing sequence {i + 1}/{min(3, len(test_loader.dataset))}, length: {len(full_sequence)}")
        max_seq_length = 30
        with torch.no_grad():
            step = 10
            for t in tqdm(range(step, len(full_sequence) + step, step)):
                if t > len(full_sequence):
                    t = len(full_sequence)
                start = max(0, t - max_seq_length)
                seq_up_to_t = full_sequence[start:t + 1].clone()

                packed_seq = pack_sequence([seq_up_to_t], enforce_sorted=False).to(device)

                # Get prediction
                output = model(packed_seq)

                softmax_output = F.softmax(output, dim=-1)
                confidence, predicted = torch.max(softmax_output, 1)

                timestep_count = t % step if t % step != 0 else step
                predicted_like_seq = np.ones(timestep_count) * predicted.item()
                confidence_like_seq = np.ones(timestep_count) * confidence.item()

                predictions.append(predicted_like_seq)
                confidences.append(confidence_like_seq)

        predictions_tensor = torch.tensor(np.hstack(predictions))
        confidences_tensor = torch.tensor(np.hstack(confidences))

        confidence_mask = confidences_tensor >= confidence_threshold

        smoothed_predictions = predictions_tensor.clone().numpy()
        smoothed_mask = confidence_mask.clone().numpy()

        for label_idx in range(len(dataset.label_dict)):
            high_conf_label_positions = np.where((smoothed_predictions == label_idx) & smoothed_mask)[0]

            if len(high_conf_label_positions) < 2:
                continue

            gaps = np.where(np.diff(high_conf_label_positions) > 1)[0]

            for gap_idx in gaps:
                start_pos = high_conf_label_positions[gap_idx] + 1
                end_pos = high_conf_label_positions[gap_idx + 1] - 1

                max_gap_size = 30
                if end_pos - start_pos > max_gap_size:
                    continue

                if not np.any(smoothed_mask[start_pos:end_pos + 1]):
                    smoothed_predictions[start_pos:end_pos + 1] = label_idx
                    smoothed_mask[start_pos:end_pos + 1] = True

        subsampled_true_labels = true_labels

        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        markers = ['o', 's', 'D', '^', 'v', 'x']  # Different markers for each label type

        ax1.set_title('Ground Truth Classification')
        ax1.plot(
            lat_long_data[:len(predictions_tensor), 0].numpy(),
            lat_long_data[:len(predictions_tensor), 1].numpy(),
            '-', color='lightgray', alpha=0.5, linewidth=0.7, label='Trajectory'
        )
        true_label_indices = subsampled_true_labels.numpy()
        for label_idx in range(len(dataset.label_dict)):
            mask = true_label_indices == label_idx
            if np.any(mask):
                label_name = reverse_label_dict[label_idx]
                ax1.plot(
                    lat_long_data[mask, 0].numpy(),
                    lat_long_data[mask, 1].numpy(),
                    markers[label_idx], label=label_name, markersize=1, markevery=(0.02, 0.02)
                )

        ax2.set_title('Model Classification')

        ax2.plot(
            lat_long_data[:len(predictions_tensor), 0].numpy(),
            lat_long_data[:len(predictions_tensor), 1].numpy(),
            '-', color='lightgray', alpha=0.5, linewidth=0.7, label='Trajectory'
        )

        for label_idx in range(len(dataset.label_dict)):
            mask = (smoothed_predictions == label_idx) & smoothed_mask
            if np.any(mask):
                label_name = reverse_label_dict[label_idx]
                ax2.plot(
                    lat_long_data[mask, 0].numpy(),
                    lat_long_data[mask, 1].numpy(),
                    markers[label_idx], label=label_name, markersize=1, markevery=(0.02, 0.02)
                )

        for ax in [ax1, ax2]:
            ax.set_xlabel(r'Latitude ($^\circ$)')
            ax.set_ylabel(r'Longitude ($^\circ$)')
            ax.set_aspect('equal')
            ax.legend(fontsize=7)
            ax.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{save_name}_sequence_{seq_id}_timestep_predictions.png", dpi=300)
        plt.close()

        if alt_idx is not None:  # Only create altitude plot if altitude column is available
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

            time_values = np.arange(len(subsampled_true_labels))

            ax1.plot(
                time_values[:len(predictions_tensor)],
                full_sequence[:len(predictions_tensor), alt_idx].numpy(),
                '-', color='lightgray', alpha=0.5, linewidth=0.7, label='Altitude Profile'
            )
            for label_idx in range(len(dataset.label_dict)):
                mask = true_label_indices == label_idx
                if np.any(mask):
                    label_name = reverse_label_dict[label_idx]
                    ax1.plot(
                        time_values[mask],
                        full_sequence[mask, alt_idx].numpy(),
                        markers[label_idx], label=label_name, markersize=2, markevery=0.01
                    )

            ax2.plot(
                time_values[:len(predictions_tensor)],
                full_sequence[:len(predictions_tensor), alt_idx].numpy(),
                '-', color='lightgray', alpha=0.5, linewidth=0.7, label='Altitude Profile'
            )

            for label_idx in range(len(dataset.label_dict)):
                mask = (smoothed_predictions == label_idx) & smoothed_mask
                if np.any(mask):
                    label_name = reverse_label_dict[label_idx]
                    ax2.plot(
                        time_values[mask],
                        full_sequence[mask, alt_idx].numpy(),
                        markers[label_idx], label=label_name, markersize=2, markevery=0.01
                    )

            for ax in [ax1, ax2]:
                ax.set_ylabel('Altitude (ft)')
                ax.legend(fontsize=7)
                ax.grid(True, linestyle='--', alpha=0.7)

            ax2.set_xlabel('Time Step (n)')

            plt.tight_layout()
            plt.savefig(f"{save_name}_sequence_{seq_id}_altitude_time.png", dpi=300)
            plt.close()


def test(model, test_loader, model_name, criterion):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    if model_name[-3:] == 'pth':
        file_name = model_name
        save_name = model_name[:-3].split('/')[-1]
    else:
        file_name = model_name + '.pth'
        save_name = model_name.split('/')[-1]
    model.load_state_dict(torch.load(file_name))
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.atleast_3d(labels)
            outputs = model(inputs)
            loss = criterion(outputs.unsqueeze(-1), torch.argmax(labels, dim=1))
            test_loss += loss.item() * len(inputs)
            _, predicted = torch.max(F.softmax(outputs, dim=-1), 1)
            correct += (predicted == torch.argmax(labels, dim=1).squeeze()).sum().item()
            total += len(labels)
            y_true.extend(torch.argmax(labels, dim=1).squeeze().cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = correct / total

    print(f'Model checkpoint {save_name} Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    class_names = list(test_loader.dataset.dataset.label_dict.keys())
    plot_confusion_matrix(y_true, y_pred, class_names)
    plt.savefig(f'{save_name}_confusion_matrix.png')
    plt.close('all')


def main(args):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    torch.set_default_dtype(torch.float32)

    if args.selected_columns is None:
        args.selected_columns = ['Altitude', 'IndicatedAirspeed', 'HeadingIndicator']#, 'Latitude', 'Longitude']

    # Load data using the data_loader
    train_loader, val_loader, test_loader = get_train_val_test(
        args.data_dir,
        batch_size=args.batch_size,
        selected_columns=args.selected_columns,
        include_derivatives=args.include_derivatives,
        derivatives_only=args.derivatives_only,
        augment_with_partials=args.augment_with_partials,
        partial_sequence_count=args.partial_sequence_count
    )

    # Calculate the input size based on feature configuration
    input_size = len(args.selected_columns)
    if args.include_derivatives:
        input_size *= 2  # Double the input size when including derivatives
    elif args.derivatives_only:
        input_size = input_size  # Input size stays the same, just using derivatives instead

    num_classes = len(train_loader.dataset.dataset.label_dict)

    if args.transformer:
        model = TransformerClassifier(input_size=input_size, lstm_hidden_size=args.lstm_n,
                                      dense_hidden_size=args.dense_n, num_classes=num_classes, args=args)
    else:
        raise NotImplementedError("LSTM model is not implemented in this script.")

    epochs, acc_threshold_met, criterion = train_model(model, train_loader, val_loader, num_epochs=args.epochs, patience=args.patience, args=args, device=device)

    test(model, test_loader, 'best_model.pth', criterion)
    # test each of the epoch checkpoints
    for epoch in range(10, epochs, 10):
        test(model, test_loader, f'checkpoint_epoch_{epoch}.pth', criterion)
    test_timestep_classification(model, test_loader, 'best_model.pth', device)
    if acc_threshold_met:
        test(model, test_loader, 'acc_threshold.pth', criterion)
        test_timestep_classification(model, test_loader, 'acc_threshold.pth', device)

if __name__ == '__main__':
    np.random.seed(42)
    args = get_args()
    torch.manual_seed(0)
    main(args)
