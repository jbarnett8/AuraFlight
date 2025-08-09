import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence


class AircraftManeuverDataset(Dataset):
    def __init__(self, directory, selected_columns=None, include_derivatives=True, derivatives_only=False, 
                 augment_with_partials=False, partial_sequence_count=3):
        self.data = []
        self.full_sequences = []
        self.full_sequences_lat_long = []  # New attribute to store latitude and longitude
        self.timestep_labels = []  # New array to store labels for every timestep
        self.labels = []
        self.sequence_id = {}
        self.file_id = {}  # Track which file each data point comes from
        self.id_2_idx = {}
        self.selected_columns = selected_columns
        self.include_derivatives = include_derivatives
        self.derivatives_only = derivatives_only
        self.num_classes = 0
        self.label_dict = {}
        self.augment_with_partials = augment_with_partials
        self.partial_sequence_count = partial_sequence_count
        self.load_data(directory)

    def resample_to_1hz(self, data, timestamps):
        """
        Resample data to 1 Hz using linear interpolation.
        
        Args:
            data (numpy.ndarray): Input data array of shape (time_steps, features)
            timestamps (numpy.ndarray): Timestamps in milliseconds
            
        Returns:
            numpy.ndarray: Resampled data at 1 Hz
        """
        # Convert milliseconds to seconds
        timestamps_sec = timestamps / 1000.0
        
        # Get start and end times, rounded to nearest seconds
        start_time = int(np.ceil(timestamps_sec[0]))
        end_time = int(np.floor(timestamps_sec[-1]))
        
        # Create regular time points at 1 Hz
        regular_timestamps = np.arange(start_time, end_time + 1)
        
        # Initialize array for resampled data
        resampled_data = np.zeros((len(regular_timestamps), data.shape[1]))
        
        # Perform linear interpolation for each feature
        for i in range(data.shape[1]):
            resampled_data[:, i] = np.interp(regular_timestamps, timestamps_sec, data[:, i])
        
        return resampled_data
    
    def resample_labels_to_1hz(self, timestamps, labels):
        """
        Resample labels to match 1 Hz timestamps using nearest neighbor interpolation.
        
        Args:
            timestamps (numpy.ndarray): Original timestamps in milliseconds
            labels (list or array): Original labels corresponding to timestamps
            
        Returns:
            list: Resampled labels at 1 Hz
        """
        # Convert milliseconds to seconds
        timestamps_sec = timestamps / 1000.0
        
        # Get start and end times, rounded to nearest seconds
        start_time = int(np.ceil(timestamps_sec[0]))
        end_time = int(np.floor(timestamps_sec[-1]))
        
        # Create regular time points at 1 Hz
        regular_timestamps = np.arange(start_time, end_time + 1)
        
        # For each regular timestamp, find the nearest original timestamp
        resampled_labels = []
        for ts in regular_timestamps:
            # Find the index of the closest timestamp
            idx = np.abs(timestamps_sec - ts).argmin()
            resampled_labels.append(labels[idx])
            
        return resampled_labels

    def calculate_derivatives(self, data):
        """
        Calculate time derivatives of data.
        
        Args:
            data (numpy.ndarray): Input data array of shape (time_steps, features)
            
        Returns:
            numpy.ndarray: Derivatives of shape (time_steps, features)
        """
        # Calculate differences between consecutive time steps
        derivatives = np.zeros_like(data)
        derivatives[1:] = data[1:] - data[:-1]
        # First row derivative is set to 0 (since we don't have previous data)
        return derivatives

    def generate_partial_sequences(self, data, label, min_percentage=0.4):
        """
        Generate partial sequences from a full maneuver sequence.
        
        Args:
            data (numpy.ndarray): Full sequence data
            label: Label for the sequence
            min_percentage (float): Minimum percentage of the sequence to include
            
        Returns:
            list: List of (partial_data, label) pairs
        """
        seq_length = len(data)
        if seq_length < 5:  # Skip very short sequences
            return [(data, label)]
            
        partials = [(data, label)]  # Always include the full sequence
        rng = np.random.default_rng(seed=42)
        lower_bound = max(10, int(seq_length * min_percentage))
        end_idx_choices = rng.choice(np.arange(lower_bound, seq_length),
                                     size=min(self.partial_sequence_count, seq_length - lower_bound),
                                     replace=False)

        for end_idx in end_idx_choices:
            partial_data = data[:end_idx]
            partials.append((partial_data, label))
            
        return partials

    def load_data(self, directory):
        all_labels = []
        seg_id = 0
        file_id = 0  # Track file ID separately
        
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath, dtype={0: str})
                
                # Find Milliseconds column index before filtering columns
                ms_col_idx = -1
                for i, col in enumerate(df.columns[1:], 1):  # Start from 1 to skip marker column
                    if "Milliseconds" in col:
                        ms_col_idx = i
                        break
                
                if ms_col_idx == -1:
                    raise ValueError(f"Milliseconds column not found in {filename}")
                
                # Find Latitude and Longitude column indices
                lat_col_idx = -1
                lon_col_idx = -1
                for i, col in enumerate(df.columns[1:], 1):  # Start from 1 to skip marker column
                    if "Latitude" in col:
                        lat_col_idx = i
                        break
                
                for i, col in enumerate(df.columns[1:], 1):  # Start from 1 to skip marker column
                    if "Longitude" in col:
                        lon_col_idx = i
                        break
                    
                # Extract timestamps before filtering columns
                timestamps = df.iloc[:, ms_col_idx].values
                
                # Extract latitude and longitude data if found
                lat_long_data = None
                if lat_col_idx != -1 and lon_col_idx != -1:
                    lat_data = df.iloc[:, lat_col_idx].values
                    lon_data = df.iloc[:, lon_col_idx].values
                    lat_long_data = np.column_stack((lat_data, lon_data))
                    
                    # Resample lat/long data to 1 Hz
                    lat_long_data = self.resample_to_1hz(lat_long_data.reshape(-1, 2), timestamps)
                    
                # Now filter columns if needed
                if self.selected_columns:
                    # Ensure the first column is always included
                    first_column = df.columns[0]
                    selected_columns = [first_column] + [col for col in self.selected_columns if col in df.columns]
                    df = df[selected_columns]
                
                # Prepare data excluding the marker column
                full_data = df.iloc[:, 1:].values
                
                # Resample full data to 1 Hz
                resampled_full_data = self.resample_to_1hz(full_data, timestamps)
                
                # Get labels for every timestep
                original_timestep_labels = self.get_timestep_labels(df)
                
                # Resample labels to 1 Hz
                resampled_timestep_labels = self.resample_labels_to_1hz(timestamps, original_timestep_labels)

                # Unwrap the heading if it exists
                if 'HeadingIndicator' in self.selected_columns:
                    print('*** NOTICE ***: UNWRAPPING HEADING INDICATOR')
                    heading_col_idx = self.selected_columns.index('HeadingIndicator')
                    resampled_full_data[:, heading_col_idx] = np.unwrap(resampled_full_data[:, heading_col_idx], period=360)
                
                # Calculate derivatives if needed for full sequence
                if self.include_derivatives or self.derivatives_only:
                    derivatives = self.calculate_derivatives(resampled_full_data)
                    
                    if self.derivatives_only:
                        processed_data = derivatives
                    else:
                        processed_data = np.hstack((resampled_full_data, derivatives))
                else:
                    processed_data = resampled_full_data
                
                self.full_sequences.append(torch.tensor(processed_data, dtype=torch.float32))
                if lat_long_data is not None:
                    self.full_sequences_lat_long.append(torch.tensor(lat_long_data, dtype=torch.float32))
                else:
                    # If lat/long not found, add empty tensor to maintain alignment
                    self.full_sequences_lat_long.append(torch.zeros((len(resampled_full_data), 2), dtype=torch.float32))
                    
                self.timestep_labels.append(resampled_timestep_labels)
                
                # Process labels and split data into segments
                segments = self.split_by_maneuver(df, timestamps)
                for segment_data, segment_times, segment_label in segments:
                    # Resample segment data to 1 Hz
                    resampled_segment_data = self.resample_to_1hz(segment_data, segment_times)
                    
                    # Calculate derivatives for this segment if needed
                    if self.include_derivatives or self.derivatives_only:
                        segment_derivatives = self.calculate_derivatives(resampled_segment_data)

                        if self.derivatives_only:
                            segment_processed_data = segment_derivatives
                        else:
                            segment_processed_data = np.hstack((resampled_segment_data, segment_derivatives))
                    else:
                        segment_processed_data = resampled_segment_data

                    # Generate partial sequences if augmentation is enabled
                    if self.augment_with_partials:
                        segment_sequences = self.generate_partial_sequences(segment_processed_data, segment_label)
                    else:
                        segment_sequences = [(segment_processed_data, segment_label)]

                    # Add all sequences (full and partial if augmentation is enabled)
                    for seq_data, seq_label in segment_sequences:
                        seq_tensor = torch.tensor(seq_data, dtype=torch.float32)
                        if len(seq_tensor) > 0:  # Only add non-empty sequences
                            self.data.append(seq_tensor)
                            self.labels.append(seq_label)
                            self.sequence_id[len(self.labels) - 1] = seg_id
                            self.file_id[len(self.labels) - 1] = file_id
                            if seg_id not in self.id_2_idx:
                                self.id_2_idx[seg_id] = []
                            self.id_2_idx[seg_id].append(len(self.labels) - 1)
                            all_labels.append(seq_label)
                    seg_id += 1
                file_id += 1  # Increment file ID for each new file

        unique_labels = np.unique(all_labels)
        self.label_dict = {label: i for i, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        self.labels = [F.one_hot(torch.tensor(self.label_dict[label]), num_classes=self.num_classes).long()
                       for label in self.labels]

        # Apply the same one-hot encoding to timestep labels
        for i in range(len(self.timestep_labels)):
            labels_tensor = torch.tensor([self.label_dict.get(label, self.label_dict.get("UNLABELLED", -1))
                                          for label in self.timestep_labels[i]])
            self.timestep_labels[i] = labels_tensor

    def get_timestep_labels(self, df):
        """
        Create labels for every timestep in the dataframe.

        Args:
            df (pd.DataFrame): The input data frame.

        Returns:
            list: A list of labels for each timestep.
        """
        timestep_labels = ["UNLABELLED"] * len(df)
        current_label = "UNLABELLED"

        for idx, marker in enumerate(df.iloc[:, 0]):  # First column contains markers
            if pd.notna(marker):
                marker = str(marker)
                if "START" in marker[-5:]:
                    current_label = ' '.join(marker.split()[:-1])
                elif "END" in marker[-3:]:
                    current_label = "UNLABELLED"

            timestep_labels[idx] = current_label

        return timestep_labels

    def split_by_maneuver(self, df, timestamps):
        """
        Splits the data into segments based on maneuvers.

        Args:
            df (pd.DataFrame): The input data frame.
            timestamps (numpy.ndarray): The timestamps for each row.

        Returns:
            list: A list of tuples (data, timestamps, label) for each maneuver segment.
        """
        segments = []
        current_label = "UNLABELLED"
        start_idx = 0

        for idx, marker in enumerate(df.iloc[:, 0]):  # First column contains markers
            if pd.notna(marker):
                marker = str(marker)
                if "START" in marker[-5:]:
                    if current_label != "UNLABELLED":
                        # Save the previous segment
                        segment_data = df.iloc[start_idx:idx, 1:].values  # Exclude the first column (MARKER)
                        segment_times = timestamps[start_idx:idx]
                        segments.append((segment_data, segment_times, current_label))
                    # Update the current label and start index
                    current_label = ' '.join(marker.split()[:-1])
                    start_idx = idx
                elif "END" in marker[-3:]:
                    if current_label != "UNLABELLED":
                        # Save the current segment
                        segment_data = df.iloc[start_idx:idx + 1, 1:].values  # Exclude the first column (MARKER)
                        segment_times = timestamps[start_idx:idx + 1]
                        segments.append((segment_data, segment_times, current_label))
                    # Reset to UNLABELLED
                    current_label = "UNLABELLED"
                    start_idx = idx + 1

        # Handle the last segment if it wasn't closed
        if current_label != "UNLABELLED" and start_idx < len(df):
            segment_data = df.iloc[start_idx:, 1:].values  # Exclude the first column (MARKER)
            segment_times = timestamps[start_idx:]
            segments.append((segment_data, segment_times, current_label))

        return segments

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_full_sequence(self, idx):
        """
        Get a full sequence and its timestep labels.

        Args:
            idx (int): Index of the sequence to retrieve

        Returns:
            tuple: (data, labels) where data and labels are tensors
        """
        seq_id = self.sequence_id[idx]
        file_idx = self.file_id[idx]  # Get the correct file index
        
        # Since we've resampled everything to 1 Hz, we can just return the full sequence
        full_seq = self.full_sequences[file_idx]
        full_labels = self.timestep_labels[file_idx]

        return full_seq, full_labels

def collate_fn(batch):
    data, labels = zip(*batch)
    data_packed = pack_sequence(data, enforce_sorted=False)  # Pack the sequences
    labels = torch.stack([v for v in labels]).to(dtype=torch.long)  # Convert labels to tensor
    return data_packed, labels

def get_train_val_test(directory, fractions=None, batch_size=10, shuffle=True, selected_columns=None,
                      include_derivatives=True, derivatives_only=False, augment_with_partials=False,
                      partial_sequence_count=3):
    if fractions is None:
        fractions = [0.7, 0.15, 0.15]
    dataset = AircraftManeuverDataset(
        directory,
        selected_columns,
        include_derivatives,
        derivatives_only,
        augment_with_partials=augment_with_partials,
        partial_sequence_count=partial_sequence_count
    )
    if len(dataset) == 0:
        raise ValueError("No data found. Please check the directory and ensure it contains .csv files.")

    # Get all unique file IDs
    unique_file_ids = sorted(np.unique([dataset.file_id[i] for i in range(len(dataset))]))
    
    # Split the files (not individual data points) into train/val/test
    train_files, temp_files = train_test_split(
        unique_file_ids, 
        test_size=(fractions[1] + fractions[2]), 
        random_state=42
    )
    
    val_files, test_files = train_test_split(
        temp_files, 
        test_size=fractions[2] / (fractions[1] + fractions[2]), 
        random_state=42
    )
    
    # Create indices based on file membership
    train_indices = [i for i in range(len(dataset)) if dataset.file_id[i] in train_files]
    val_indices = [i for i in range(len(dataset)) if dataset.file_id[i] in val_files]
    test_indices = [i for i in range(len(dataset)) if dataset.file_id[i] in test_files]
    
    # Verify no overlap between sets
    tv = set.intersection(set(train_indices), set(val_indices))
    tt = set.intersection(set(train_indices), set(test_indices))
    vt = set.intersection(set(val_indices), set(test_indices))
    assert(len(tv) == 0 and len(tt) == 0 and len(vt) == 0), f"Train, val, and test sets overlap: {len(tv)}, {len(tt)}, {len(vt)}"

    # Create subsets
    full_sequences = dataset.full_sequences.copy()
    dataset.full_sequences = [full_sequences[v] for v in test_files]
    dataset.timestep_labels = [dataset.timestep_labels[v] for v in test_files]
    dataset.full_sequences_lat_long = [dataset.full_sequences_lat_long[v] for v in test_files]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
