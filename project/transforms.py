import emd
import numpy as np
from scipy.signal import istft, stft


class Transform:
    def __init__(self, probability=1.0, random_state=None):
        self.probability = probability
        self.rng = np.random.default_rng(random_state)
        self._indices = None

    def __call__(self, X, y):
        X_out, y_out = X.copy(), y.copy()
        mask = self._get_mask(size=len(X))
        if np.any(mask):
            self._indices = np.argwhere(mask)[:, 0]
            X_out[mask, ...], y_out[mask, ...] = self.operation(
                X_out[mask, ...], y_out[mask, ...]
            )
        return X_out, y_out

    def fit(self, X, y=None):
        pass

    def _get_mask(self, size=None):
        return self.probability >= self.rng.uniform(size=size)


class GaussianNoise(Transform):
    def __init__(self, probability, std=0.1, random_state=None):
        super().__init__(probability, random_state)
        self.std = std

    def operation(self, X, y):
        noise = self.rng.normal(loc=0.0, scale=self.std, size=X.shape)
        return X + noise, y


class TimeFrequencyGaussianNoise(Transform):
    def __init__(self, probability, sfreq, std=0.01, random_state=None):
        super().__init__(probability, random_state)
        self.sfreq = sfreq
        self.std = std
        self.window = "hann"
        self.nperseg = sfreq // 4 + 1

    def operation(self, X, y):
        _, _, Zxx = stft(X, fs=self.sfreq, window=self.window, nperseg=self.nperseg)
        amp, phase = np.abs(Zxx), np.angle(Zxx)
        noise = self.rng.normal(loc=0.0, scale=self.std, size=Zxx.shape)
        amp = amp + noise
        Zxx = amp * (np.cos(phase) + 1j * np.sin(phase))
        _, X_out = istft(Zxx, fs=self.sfreq, window=self.window, nperseg=self.nperseg)
        return X_out, y


class SlidingWindow(Transform):
    def __init__(self, probability, smin, smax, sfreq, random_state=None):
        super().__init__(probability, random_state)
        self.smin = round(smin * sfreq)
        self.smax = round(smax * sfreq)

    def operation(self, X, y):
        for i, x in enumerate(X):
            shift = self.rng.integers(self.smin, self.smax)
            X[i] = np.roll(x, shift)
        return X, y


class SlidingRecombination(Transform):
    def __init__(self, probability, smin, smax, sfreq, tmin, tmax, random_state=None):
        super().__init__(probability, random_state)
        self.sliding_window = SlidingWindow(probability, smin, smax, sfreq)
        self.recombination = TimeFrequencyRecombination(probability, sfreq)
        self.tmin = round(tmin * sfreq)
        self.tmax = round(tmax * sfreq)

    def operation(self, X, y):
        X_s, y = self.sliding_window.operation(X, y)
        X_r, y = self.recombination.operation(X_s[:, :, self.tmin : self.tmax], y)
        X[:, :, self.tmin : self.tmax] = X_r

        return X, y


class TimeFrequencyRecombination(Transform):
    def __init__(
        self, probability, sfreq, n_segments=8, recombine_freqs=False, random_state=None
    ):
        super().__init__(probability, random_state)
        self.sfreq = sfreq
        self.window = "hamming"
        self.nperseg = sfreq // 4 + 1
        self.n_segments = n_segments
        self.recombine_freqs = recombine_freqs

    def operation(self, X, y):
        _, _, Zxx = stft(X, fs=self.sfreq, window=self.window, nperseg=self.nperseg)
        Zxx = Zxx.reshape(Zxx.shape[:-1] + (self.n_segments, -1))

        n_freqs, n_times = Zxx.shape[-3:-1]
        for l in np.unique(y):
            idx = np.argwhere(y == l)[:, 0]
            # Recombine time windows
            for t in range(n_times):
                new_idx = self.rng.permutation(idx)
                Zxx[idx, :, :, t, :] = Zxx[new_idx, :, :, t, :]
            # Recombine frequency windows
            if self.recombine_freqs:
                for f in range(n_freqs):
                    new_idx = self.rng.permutation(idx)
                    Zxx[idx, :, f, :, :] = Zxx[new_idx, :, f, :, :]

        Zxx = Zxx.reshape(Zxx.shape[:-2] + (-1,))
        _, X_out = istft(Zxx, fs=self.sfreq, window=self.window, nperseg=self.nperseg)

        return X_out, y


class EMD(Transform):
    """WARNING: this transform only works when applied to the exact same X as the one it was fit to."""

    def __init__(self, probability, random_state=None):
        super().__init__(probability, random_state)
        self.imfs = None

    def operation(self, X, y):
        imfs = self.imfs[self._indices]
        X_out = X - np.sum(imfs, axis=-1)
        for l in np.unique(y):
            idx = np.argwhere(y == l)[:, 0]
            n_imfs = imfs.shape[-1]
            for i in range(n_imfs):
                new_idx = self.rng.permutation(idx)
                imfs[idx, :, :, i] = imfs[new_idx, :, :, i]
        X_out = X_out + np.sum(imfs, axis=-1)

        return X_out, y

    def fit(self, X, y):
        # Calculate intrinsic mode functions for all channels of all samples
        imfs = [[emd.sift.sift(channel) for channel in sample] for sample in X]
        # Replace missing IMFs with zero
        max_imfs = max((channel.shape[-1] for sample in imfs for channel in sample))
        for i, sample in enumerate(imfs):
            for c, channel in enumerate(sample):
                n = channel.shape[-1]
                npad = [(0, 0), (0, max_imfs - n)]
                imfs[i][c] = np.pad(channel, npad, mode="constant", constant_values=0)
        self.imfs = np.array(imfs)


class Interpolation(Transform):
    def __init__(self, probability, random_state=None):
        super().__init__(probability, random_state)

    def operation(self, X, y):
        for l in np.unique(y):
            idx = np.argwhere(y == l)[:, 0]
            new_idx = self.rng.permutation(idx)
            X[idx] = (X[idx] + X[new_idx]) / 2
        return X, y
