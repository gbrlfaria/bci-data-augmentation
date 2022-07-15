#BCIC-IV 2a
# cross-session experiments
python -m project 2a --sfreq=128 --h_freq=40 --n_classes=2 --augment_probability=0.5 --transform=gaussian &&
# within-session experiments
python -m project 2a --sfreq=128 --h_freq=40 --n_classes=2 --intrasession --epochs=1000 --augment_probability=0.5 --transform=gaussian --keep_train=6  --keep_valid=3  &&
python -m project 2a --sfreq=128 --h_freq=40 --n_classes=2 --intrasession --epochs=1000 --augment_probability=0.5 --transform=gaussian --keep_train=12 --keep_valid=6  &&
python -m project 2a --sfreq=128 --h_freq=40 --n_classes=2 --intrasession --epochs=1000 --augment_probability=0.5 --transform=gaussian --keep_train=18 --keep_valid=9  &&
python -m project 2a --sfreq=128 --h_freq=40 --n_classes=2 --intrasession --epochs=1000 --augment_probability=0.5 --transform=gaussian --keep_train=24 --keep_valid=12 &&
python -m project 2a --sfreq=128 --h_freq=40 --n_classes=2 --intrasession --epochs=1000 --augment_probability=0.5 --transform=gaussian --keep_train=30 --keep_valid=15 &&
python -m project 2a --sfreq=128 --h_freq=40 --n_classes=2 --intrasession --epochs=1000 --augment_probability=0.5 --transform=gaussian &&

# BCIC-IV 2b
# cross-session experiments
python -m project 2b --sfreq=128 --h_freq=40 --augment_probability=0.5 --transform=sliding-window-x --crop_after &&
# within-session experiments
python -m project 2b --sfreq=128 --h_freq=40 --intrasession --epochs=1000 --train_size=120 --augment_probability=0.5 --transform=sliding-window-x --crop_after --keep_train=6  --keep_valid=3  &&
python -m project 2b --sfreq=128 --h_freq=40 --intrasession --epochs=1000 --train_size=120 --augment_probability=0.5 --transform=sliding-window-x --crop_after --keep_train=12 --keep_valid=6  &&
python -m project 2b --sfreq=128 --h_freq=40 --intrasession --epochs=1000 --train_size=120 --augment_probability=0.5 --transform=sliding-window-x --crop_after --keep_train=18 --keep_valid=9  &&
python -m project 2b --sfreq=128 --h_freq=40 --intrasession --epochs=1000 --train_size=120 --augment_probability=0.5 --transform=sliding-window-x --crop_after --keep_train=24 --keep_valid=12 &&
python -m project 2b --sfreq=128 --h_freq=40 --intrasession --epochs=1000 --train_size=120 --augment_probability=0.5 --transform=sliding-window-x --crop_after --keep_train=30 --keep_valid=15
