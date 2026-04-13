import argparse

def main():
    parser = argparse.ArgumentParser(description="Fully runs preprocessing pipeline using the Orchestrator.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='presidents_to_abortion', help="Name of training plan to finetune on.")
    parser.add_argument("--subproportions", type=float, nargs='+', default=[0.1, 0.2, 0.5, 0.7, 1.0], help="Subsamples of the complete training set size to actually train on.")
    parser.add_argument("--train_ratio", type=float, nargs='?', default=0.7, help="Proportion of data on training.")
    parser.add_argument("--val_ratio", type=float, nargs='?', default=0.2, help="Proportion of data on validation.")
    parser.add_argument("--test_ratio", type=float, nargs='?', default=0.1, help="Proportion of data on test.")
    parser.add_argument("--seed", type=int, nargs='?', default=42, help="Seed for any and all random processes")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True) # Set True to see the metadata
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    
    args = parser.parse_args()
    split_handler = SplitHandler( # randomness based on training setting
        train_plan=args.train_plan, 
        subproportions=args.subproportions, 
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio, 
        seed=args.seed
    )
    split_handler.subdivide_training_set(save=args.save, debug=args.debug, verbose=args.verbose)


if __name__ == "__main__":
    main()