from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--sfreq", default=250, type=int)
    parser.add_argument("--l_freq", default=4, type=int)
    parser.add_argument("--h_freq", default=None, type=int)
    parser.add_argument("--tmin", default=0.5, type=float)
    parser.add_argument("--tmax", default=2.5, type=float)
    parser.add_argument("--tstart", default=-0.5, type=float)
    parser.add_argument("--model", default="eegnet-82")
    parser.add_argument("--n_classes", default=None, type=int)
    parser.add_argument("--n_electrodes", default=None, type=int)
    parser.add_argument("--train_size", default=None, type=int)
    parser.add_argument("--transform", default=None)
    parser.add_argument("--crop_after", action="store_true", default=False)
    parser.add_argument("--augment", default=1, type=float)
    parser.add_argument("--augment_probability", default=1, type=float)
    parser.add_argument("--keep_train", default=None, type=int)
    parser.add_argument("--keep_valid", default=None, type=int)
    parser.add_argument("--intrasession", action="store_true", default=False)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    if args.n_classes is None:
        if args.dataset == "2a":
            args.n_classes = 4
        elif args.dataset == "2b":
            args.n_classes = 2
    if args.n_electrodes is None:
        if args.dataset == "2a":
            args.n_electrodes = 22
        elif args.dataset == "2b":
            args.n_electrodes = 3

    from project.experiments import runner

    runner.run(args)
