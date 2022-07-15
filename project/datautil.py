import math
import numpy as np


class TransformEpochIterator:
    def __init__(self, X, y, epoch_size, transform):
        self.X = X
        self.y = y
        self.epoch_size = epoch_size
        self.transform = transform
        self.iteration = 0

    def __iter__(self):
        return self

    def __next__(self):
        min_idx = self.iteration * self.epoch_size
        max_idx = (self.iteration + 1) * self.epoch_size

        if max_idx >= len(self.X):
            max_idx = len(self.X)
            self.iteration = 0
        else:
            self.iteration += 1

        X, y = self.X[min_idx:max_idx], self.y[min_idx:max_idx]
        if self.transform is not None:
            X, y = self.transform(X, y)
        return X, y


def time_crop(X, tmin, tmax, sfreq):
    min_idx = round(tmin * sfreq)
    max_idx = round(tmax * sfreq)
    return X[:, :, min_idx:max_idx]


def filter_data(X, y, keep):
    X_out, y_out = [], []
    for label in np.unique(y):
        X_label = X[y == label]

        indices = np.random.permutation(len(X_label))
        indices = indices[:keep]

        X_f = X_label[indices]
        y_f = np.array([label] * len(X_f))

        X_out.append(X_f)
        y_out.append(y_f)

    return np.concatenate(X_out), np.concatenate(y_out)


def augment_data(X, y, transform, size):
    num_aug = math.ceil(size / len(X)) - 1
    if num_aug == 0:
        return X, y

    X_aug, y_aug = [], []
    for _ in range(num_aug):
        X_new, y_new = transform(X, y)
        X_aug.append(X_new)
        y_aug.append(y_new)
    X_aug, y_aug = np.concatenate(X_aug), np.concatenate(y_aug)

    if len(X) + len(X_aug) > size:
        # Shuffle augmented data to avoid bias when cropping extra samples
        indices = np.random.permutation(len(X_aug))
        X_aug, y_aug = X_aug[indices], y_aug[indices]

        labels = np.unique(y_aug)
        stratified_samples = {label: [] for label in labels}
        for i in range(size - len(X)):
            l = labels[i % len(labels)]
            # Get all indices of the class
            indices = np.where(y_aug == l)[0]
            # Pick the next sample of the class
            j = len(stratified_samples[l]) % len(indices)
            stratified_samples[l].append(indices[j])

        indices = [idx for indices in stratified_samples.values() for idx in indices]
        X_aug = X_aug[indices]
        y_aug = y_aug[indices]

    return np.concatenate([X, X_aug]), np.concatenate([y, y_aug])
