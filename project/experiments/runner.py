import torch
import wandb
from braindecode import EEGClassifier, preprocessing
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor
from braindecode.util import set_random_seeds
from project import datautil
from project.datautil import TransformEpochIterator
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from skorch.callbacks import Checkpoint, WandbLogger
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from . import util

set_random_seeds(seed=42, cuda=True)


def run(args):
    if args.dataset == "2a" or args.dataset == "2ab":
        for subject_id in range(1, 10):
            run_subject(subject_id, dataset_name="BNCI2014001", args=args)
    elif args.dataset == "2b":
        for subject_id in range(1, 10):
            run_subject(subject_id, dataset_name="BNCI2014004", args=args)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset}")


def run_subject(subject_id, dataset_name, args):
    # Load
    data = MOABBDataset(dataset_name, subject_ids=[subject_id])

    # Preprocess
    preprocessors = []
    if args.dataset == "2a" and args.n_electrodes == 3:
        selection = ["C3", "Cz", "C4"]
    else:
        selection = None
    preprocessors.append(
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, selection=selection)
    )
    preprocessors.append(
        Preprocessor(preprocessing.scale, factor=1e6, apply_on_array=True)
    )
    preprocessors.append(
        Preprocessor("filter", l_freq=args.l_freq, h_freq=args.h_freq, verbose=False)
    )
    if args.sfreq != 250:
        preprocessors.append(Preprocessor("resample", sfreq=args.sfreq, verbose=False))
    preprocessors.append(
        Preprocessor(
            preprocessing.exponential_moving_standardize,
            factor_new=0.001,
            init_block_size=1000,
        )
    )
    preprocessing.preprocess(data, preprocessors)

    # Epoch
    epochs = preprocessing.create_windows_from_events(
        data, trial_start_offset_samples=round(args.tstart * args.sfreq), preload=True
    )

    if args.intrasession:
        for s, ds in enumerate(epochs.split(by="session").values()):
            cross_validate(ds, subject_id, session_id=s, args=args)
    else:
        cross_validate(epochs, subject_id, session_id=-1, args=args)


def cross_validate(epochs, subject_id, session_id, args):
    # K-Fold
    skf = StratifiedKFold(n_splits=4)
    for fold_idx, (train_index_, test_index) in enumerate(
        skf.split(*epochs[range(len(epochs))])
    ):
        # Split
        skf = StratifiedKFold(n_splits=3)
        split = skf.split(*epochs[train_index_])
        for _ in range(fold_idx % 3 + 1):
            train_index, valid_index = next(split)
        train_index = train_index_[train_index]
        valid_index = train_index_[valid_index]

        train_fold(
            epochs,
            train_index,
            valid_index,
            test_index,
            fold_idx,
            subject_id,
            session_id,
            args,
        )


def train_fold(
    epochs,
    train_index,
    valid_index,
    test_index,
    fold_idx,
    subject_id,
    session_id,
    args,
):
    # Init logging
    wandb_run = wandb.init(reinit=True, config=vars(args))

    X_train, y_train = epochs[train_index]
    X_valid, y_valid = epochs[valid_index]
    X_test, y_test = epochs[test_index]

    if args.dataset == "2a" and args.n_classes == 2:
        indices = (y_train == 1) | (y_train == 2)
        X_train, y_train = X_train[indices], y_train[indices] - 1
        indices = (y_valid == 1) | (y_valid == 2)
        X_valid, y_valid = X_valid[indices], y_valid[indices] - 1
        indices = (y_test == 1) | (y_test == 2)
        X_test, y_test = X_test[indices], y_test[indices] - 1

    # Filter
    train_size = args.train_size or len(X_train)
    if args.keep_train:
        X_train, y_train = datautil.filter_data(X_train, y_train, args.keep_train)
    if args.keep_valid is not None:
        X_valid, y_valid = datautil.filter_data(X_valid, y_valid, args.keep_valid)

    # Crop
    tmin = args.tmin - args.tstart
    tmax = args.tmax - args.tstart
    if not args.crop_after:
        X_train = datautil.time_crop(X_train, tmin, tmax, args.sfreq)
    X_valid = datautil.time_crop(X_valid, tmin, tmax, args.sfreq)
    X_test = datautil.time_crop(X_test, tmin, tmax, args.sfreq)

    # Create transform
    transform = util.get_transform(
        name=args.transform,
        probability=args.augment_probability,
        sfreq=args.sfreq,
        tmin=tmin,
        tmax=tmax,
    )
    if transform is not None:
        transform.fit(X_train, y_train)

    # Augment data (offline)
    if args.augment_probability == 1 and transform is not None:
        X_train, y_train = datautil.augment_data(
            X_train, y_train, transform, size=train_size * args.augment
        )
        transform = None

    # Create model
    n_channels, n_times = X_test.shape[1:]
    n_classes = args.n_classes
    model = util.get_model(args.model, n_channels, n_times, n_classes)

    # Update run config
    wandb_run.config.update(
        {
            "subject": subject_id,
            "session": session_id,
            "fold_idx": fold_idx + 1,
            "train_index": train_index,
            "valid_index": valid_index,
            "test_index": test_index,
            "train_len": len(X_train),
            "valid_len": len(X_valid),
            "test_len": len(X_test),
        }
    )
    wandb_run.watch(model, log="all", log_freq=100, log_graph=True)

    # Setup training
    clf = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-3,
        train_split=predefined_split(Dataset(X=X_valid, y=y_valid)),
        batch_size=args.batch_size,
        callbacks=[
            "accuracy",
            Checkpoint(
                dirname="checkpoints",
                load_best=True,
                fn_prefix=f"s_{subject_id}__f_{fold_idx + 1}_",
            ),
            WandbLogger(wandb_run),
        ],
        device=args.device,
    )
    dataset_iterator = TransformEpochIterator(
        X_train, y_train, epoch_size=train_size, transform=transform
    )

    # Fit
    clf.initialize()
    clf.notify("on_train_begin", X=X_train, y=y_train)
    for _ in range(args.epochs):
        X, y = next(dataset_iterator)
        if args.crop_after:
            X = datautil.time_crop(X, tmin, tmax, args.sfreq)
        clf.fit_loop(X, y, epochs=1)
    clf.notify("on_train_end", X=X_train, y=y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_kappa = cohen_kappa_score(y_test, y_pred)

    wandb_run.log({"test_accuracy": test_accuracy, "test_kappa": test_kappa})
