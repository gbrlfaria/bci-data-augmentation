# Analyzing Data Augmentation Methods for Convolutional Neural Network-based Brain-Computer Interfaces

*Gabriel Faria, Gabriel Henrique de Souza, Heder Bernardino, Luciana Motta, Alex Vieira*

This repository provides code, visualizations, and supplementary information about the paper.

- [:package: Installation](#installation)
- [:wrench: Usage](#usage)
- [:page_facing_up: Supplementary Material](supplementary-material.pdf)

## Installation

To install the project, run the following commands:

```bash
# Install requirements
conda env create --file=environment.yml

# Clone repository
git clone https://github.com/gabriel-dev/bci-data-augmentation
```

Alternatively, use `pip` and [`requirements.txt`](requirements.txt), which contains detailed package information.

## Usage

To run the project, execute:

```bash
python -m project <DATASET> [optional-arguments]

positional arguments:
  DATASET                          # the dataset to be used. Available options: 2a and 2b

optional arguments:
  -h, --help                       # show help message and exit
  --sfreq SFREQ                    # the sampling frequency of the signals after preprocessing. Defaults to 250 Hz
  --l_freq L_FREQ                  # lowest allowed frequency in Hertz after band-pass filtering. Defaults to 4 Hz
  --h_freq H_FREQ                  # highest allowed frequency in Hertz after band-pass filtering. Disabled by default
  --tmin TMIN                      # the start of the crop in seconds, relative to the cue onset. Defaults to 0.5
  --tmax TMAX                      # the end of the crop in seconds, relative to the cue onset. Defaults to 2.5
  --model MODEL                    # the neural network model to be used. Check experiments/util.py for available options. Defaults to eegnet-82
  --n_classes N_CLASSES            # the number of output classes to be used. Defaults to the dataset's default configuration
  --n_electrodes N_ELECTRODES      # the number of electrodes to be used. Defaults to the dataset's default configuration
  --train_size TRAIN_SIZE          # determines the max number of training samples in an epoch. Defaults to the size of the training set
  --transform TRANSFORM            # the data augmentation method to be used. Check experiments/util.py for available options. Defaults to none
  --crop_after                     # crop the trials only after applying data augmentation. Necessary for Sliding Window and SW+SR. Disabled by default
  --augment_probability PROBA      # probability of the transform being applied to a training sample during an epoch. Defaults to 1
  --keep_train KEEP_TRAIN          # the number training samples per class to keep (the rest is ignored). Defaults to the maximum
  --keep_valid KEEP_VALID          # the number validation samples per class to keep (the rest is ignored). Defaults to the maximum
  --intrasession                   # use data from one session at a time (within-session). By default, all available data is used (cross-session)
  --batch_size BATCH_SIZE          # sets the batch size. Defaults to 32
  --epochs EPOCHS                  # sets the number of training epochs. Defaults to 500
  --device DEVICE                  # sets which PyTorch device to use. Defaults to "cuda"
```

An example of a run performed during our experiments is shown in [`run-example.sh`](run-example.sh).

The results obtained in our experiments are available in [`data/results.csv`](data/results.csv).
The accuracy and loss curve history of the experiments is available [here](https://drive.google.com/file/d/152dnBfciU1M0pbjrIP20VVo02prpssb-/view).
