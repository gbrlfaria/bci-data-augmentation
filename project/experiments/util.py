from project.transforms import (
    EMD,
    GaussianNoise,
    Interpolation,
    SlidingRecombination,
    SlidingWindow,
    TimeFrequencyGaussianNoise,
    TimeFrequencyRecombination,
)
from project.models.eeginception import EEGInception
from project.models.eegnet import EEGNet


def get_transform(name, probability, sfreq, tmin, tmax):
    if name is None:
        return None
    if name == "gaussian":
        return GaussianNoise(probability)
    if name == "stft-gaussian-2":
        return TimeFrequencyGaussianNoise(probability, sfreq, std=0.01)
    if name == "amplitude-perturbation-15":
        return TimeFrequencyGaussianNoise(probability, sfreq, std=0.015)
    if name == "amplitude-perturbation-20":
        return TimeFrequencyGaussianNoise(probability, sfreq, std=0.020)
    if name == "sliding-window":
        return SlidingWindow(probability, smin=-1, smax=1, sfreq=sfreq)
    if name == "sliding-window-x":
        return SlidingWindow(probability, smin=-1.5, smax=1, sfreq=sfreq)
    if name == "stft-recombination":
        return TimeFrequencyRecombination(probability, sfreq)
    if name == "sliding-recombination":
        return SlidingRecombination(
            probability, smin=-1.5, smax=1, sfreq=sfreq, tmin=tmin, tmax=tmax
        )
    if name == "emd":
        return EMD(probability, sfreq)
    if name == "interpolation":
        return Interpolation(probability)

    raise ValueError(f"The transform `{name}` is invalid")


def get_model(name, n_channels, n_times, n_classes):
    if name == "eegnet-big":
        m = EEGNet(
            n_channels,
            n_times,
            n_classes,
            F1=64,
            D=4,
            F2=256,
            kernel_length=n_times // 2 // 4,
        )
        import torchinfo
        torchinfo.summary(m)
        return m
    if name == "eegnet-82":
        # time = 2 seconds
        # l_freq = 4 Hz
        return EEGNet(n_channels, n_times, n_classes, kernel_length=n_times // 2 // 4)
    if name == "eegnet-42":
        # time = 2 seconds
        # l_freq = 4 Hz
        return EEGNet(
            n_channels, n_times, n_classes, kernel_length=n_times // 2 // 4, F1=4, F2=8
        )
    if name == "eeginception":
        return EEGInception(n_channels, n_times, n_classes)

    raise ValueError(f"Invalid model name: {name}")
