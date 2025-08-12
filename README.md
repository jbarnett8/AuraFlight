# AuraFlight
The following repository contains some of the code and data used in the paper "Aura: An Automated System for the Real-Time Evaluation of Flight Maneuver Performance".

## Overview
- Automated classification of flight maneuvers using transformer-based deep learning
- Real-time evaluation capabilities for flight performance
- Visualization tools for maneuver classification and confidence metrics
- Support for both complete flight sequences and individual maneuvers

## Requirements
The project requires Python 3.9+ and the following dependencies:
```bash
# Core packages
numpy>=2.2.0
torch>=2.6.0
pandas>=2.2.3
scikit-learn>=1.6.1

# Visualization packages
matplotlib>=3.9.3
seaborn>=0.13.2

# Utilities
tqdm>=4.67.1
```

## Installation

Create a conda environment with the necessary dependencies:

```bash
conda create -n auraflight python=3.13
conda activate auraflight
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm
```

## Usage

### Reproducing Paper Results

To generate the model and synthetic test data figures presented in the paper:

```bash
./run_script.sh
```

This script:
- Parses the flight maneuver database from the `csv` directory
- Trains a Transformer-based classifier to predict individual maneuver labels
- Evaluates performance on full flight sequences

The results used in the paper are based on the best validation loss achieved during training. All model checkpoints are saved for reference.
Note that some differences may occur due to hardware and software variations despite the random seed being fixed. Results reported on the paper were obtained on a Macbook Air M1 (2020) using `torch.device("mps")`.

### Custom Experiments

The main classification script supports various command line arguments:

```bash
python classify_labelled.py [options]
```

## Data

The repository includes:
- `csv/`: Directory containing the Microsoft Flight simulator flight maneuver database in CSV format
- `augmented_filtered_data.npy`: Flight data from the Aura system augmented with G1000 latitude/longitude logs
- `column_names.txt`: Descriptions of data columns

## Output Files

Running the `make_plot.sh` script generates:
- `prediction_altitude_time_predictions.png`: True and predicted classification along the altitude plot
- `prediction_confidence.png`: Model confidence in its predictions as a function of time
- `prediction_lat_lon_predictions.png`: Predicted classification along the latitude and longitude plot

## Additional Code
Because some of the code used to generate results in the paper rely on access to specialized hardware (the OAK-D camera system, specifically), we have not provided these files to avoid confusion. However, they, along with the image data, are available upon request. 

## Citation

If you use this code or data in your research, please cite our paper:

```
@article{
TBD
}
```
