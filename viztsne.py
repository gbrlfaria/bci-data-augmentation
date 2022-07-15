import functools

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import pyriemann
import seaborn as sns
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor
from braindecode.preprocessing import (
    create_windows_from_events as bd_create_windows_from_events,
)
from braindecode.preprocessing import (
    exponential_moving_standardize as bd_exponential_moving_standardize,
)
from braindecode.preprocessing import preprocess as bd_preprocess
from sklearn.manifold import TSNE

from project import datautil
from project.experiments import util as exputil

SFREQ = 128
TSTART = -0.5
TMIN = 0.5
TMAX = 2.5


def main():
    dataset_names = ["BNCI2014001", "BNCI2014004"]
    for dataset_name in dataset_names:
        for subject_id in range(1, 9 + 1):
            X, Y = load_data(dataset_name, subject_id, preprocess=True)
            visualize_data(X, Y, name_prefix=f"d{dataset_name}_s{subject_id}_p")
            X, Y = load_data(dataset_name, subject_id, preprocess=False)
            visualize_data(X, Y, name_prefix=f"d{dataset_name}_s{subject_id}")


def load_data(dataset_name: str, subject_id: int, preprocess: bool):
    data = MOABBDataset(dataset_name, subject_ids=[subject_id])

    # Preprocess
    if preprocess:
        preprocessors = [
            Preprocessor(
                "pick_types",
                eeg=True,
                meg=False,
                stim=False,
            ),
            Preprocessor(functools.partial(lambda x: x * 1e6), apply_on_array=True),
            Preprocessor(
                "filter",
                l_freq=4,
                h_freq=40,
                verbose=False,
            ),
            Preprocessor("resample", sfreq=SFREQ, verbose=False),
            Preprocessor(
                bd_exponential_moving_standardize,
                factor_new=0.001,
                init_block_size=1000,
            ),
        ]
    else:
        preprocessors = [
            Preprocessor(
                "pick_types",
                eeg=True,
                meg=False,
                stim=False,
            ),
            Preprocessor(functools.partial(lambda x: x * 1e6), apply_on_array=True),
            Preprocessor("resample", sfreq=SFREQ, verbose=False),
            Preprocessor(
                bd_exponential_moving_standardize,
                factor_new=0.001,
                init_block_size=1000,
            ),
        ]
    bd_preprocess(data, preprocessors)

    # Epoch
    epochs = bd_create_windows_from_events(
        data, trial_start_offset_samples=round(TSTART * SFREQ), preload=True
    )
    X, Y = epochs[range(len(epochs))]

    print(dataset_name, subject_id, X.shape, np.mean(X), np.std(X))

    return X, Y


def visualize_data(X: np.ndarray, Y: np.ndarray, name_prefix: str):
    transformed_data = get_transformed_data(X, Y)

    # Join
    X_all = np.concatenate([X for X, _ in transformed_data.values()])
    Y_all = np.concatenate([Y for _, Y in transformed_data.values()])
    N = []
    for transform_name in transformed_data.keys():
        N.extend([transform_name] * len(X))
    N = np.array(N)

    # t-SNE
    pair_dist = calculate_pair_dist(X_all)
    X_tsne = tsne(pair_dist, dim=2)
    plot_tsne(X_tsne, Y_all, N, name_prefix)


def calculate_pair_dist(X: np.ndarray):
    covs = pyriemann.estimation.Covariances().transform(X)
    return pyriemann.stats.pairwise_distance(covs, metric="riemann")


def get_transformed_data(X: np.ndarray, Y: np.ndarray):
    def transform_data(X: np.ndarray, Y: np.ndarray, transform_name: str):
        crop_after = transform_name in ["sliding-window-x", "sliding-recombination"]
        tmin = TMIN - TSTART
        tmax = TMAX - TSTART

        X_in, Y_in = X, Y
        if not crop_after:
            X_in = datautil.time_crop(X_in, tmin, tmax, SFREQ)

        transform = exputil.get_transform(
            name=transform_name,
            probability=1.0,
            sfreq=SFREQ,
            tmin=tmin,
            tmax=tmax,
        )
        if transform is not None:
            transform.fit(X_in, Y_in)
            X_out, Y_out = datautil.augment_data(
                X_in, Y_in, transform=transform, size=2 * len(X_in)
            )
            X_out, Y_out = X_out[len(X_in) :], Y_out[len(Y_in) :]
        else:
            X_out, Y_out = X_in, Y_in

        if crop_after:
            X_out = datautil.time_crop(X_out, tmin, tmax, SFREQ)

        return (X_out, Y_out)

    transform_names = [
        None,
        "gaussian",
        "amplitude-perturbation-15",
        "sliding-window-x",
        "stft-recombination",
        "sliding-recombination",
        "emd",
    ]
    return {
        str(transform_name): transform_data(X, Y, transform_name)
        for transform_name in transform_names
    }


def tsne(pair_dist: np.ndarray, dim: int = 2, random_state = 42):
    X_out = TSNE(
        n_components=dim,
        metric="precomputed",
        learning_rate="auto",
        random_state=random_state
    ).fit_transform(pair_dist)
    return X_out


def plot_tsne(X: np.ndarray, Y: np.ndarray, N: np.ndarray, name_prefix: str):
    # All
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=Y, style=N, ax=ax, palette="tab10", s=64)
    ax.legend(bbox_to_anchor=(1.01, 1.01))
    fig.savefig(f"data/da/tsne_{name_prefix}_all.pdf", bbox_inches="tight")

    # Segmented
    names = set(N)
    fig, axs = plt.subplots(
        nrows=int(np.ceil(len(names) / 2)),
        ncols=2,
        figsize=(12, 16),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, name in enumerate(names):
        ax = axs[i // 2][i % 2]
        X_ = X[N == name]
        Y_ = Y[N == name]
        sns.scatterplot(x=X_[:, 0], y=X_[:, 1], hue=Y_, ax=ax, palette="tab10", s=64)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(f"data/da/tsne_{name_prefix}.pdf", bbox_inches="tight")


def plot_tsne_3d(X: np.ndarray, Y: np.ndarray, N: np.ndarray, name_prefix: str):
    names = set(N)
    labels = set(Y)

    # All
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(projection="3d")
    for i, name in enumerate(names):
        marker = ["o", "X", "s", "P", "D", "d", "^"][i]
        for j, label in enumerate(labels):
            color = ["tab:blue", "tab:orange", "tab:green", "tab:red"][j]
            X_ = X[(N == name) & (Y == label)]
            Y_ = Y[(N == name) & (Y == label)]
            ax.scatter(X_[:, 0], X_[:, 1], X_[:, 2], color=color, marker=marker, s=64)
    ax.legend(bbox_to_anchor=(1.01, 1.01))
    plt.show()
    # fig.savefig(f"data/da/tsne_3d_{name_prefix}_all.pdf", bbox_inches="tight")

    # Segmented
    fig, axs = plt.subplots(
        nrows=int(np.ceil(len(names) / 2)),
        ncols=2,
        figsize=(12, 16),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, name in enumerate(names):
        ax = axs[i // 2][i % 2]
        for j, label in enumerate(labels):
            color = ["tab:blue", "tab:orange", "tab:green", "tab:red"][j]
            X_ = X[(N == name) & (Y == label)]
            Y_ = Y[(N == name) & (Y == label)]
            ax.scatter(X_[:, 0], X_[:, 1], X_[:, 2], color=color, marker="o", s=64)
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(f"data/da/tsne_3d_{name_prefix}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
