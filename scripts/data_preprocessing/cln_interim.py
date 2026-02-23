import argparse
from calyapo.data_preprocessing.split_handler import SplitHandler

def main():
    parser = argparse.ArgumentParser(description="Build and clean raw CSVs into JSON files.")
    parser.add_argument("--dataset_name", type=str, nargs='?', default='IGS', help="Name of dataset to clean.")
    parser.add_argument("--train_plan", type=str, nargs='?', default='ideology_to_ideology', help="Name of training plan to finetune on.")
    parser.add_argument("--train_ratio", type=float, nargs='?', default=0.7, help="Proportion of data on training.")
    parser.add_argument("--val_ratio", type=float, nargs='?', default=0.2, help="Proportion of data on validation.")
    parser.add_argument("--test_ratio", type=float, nargs='?', default=0.1, help="Proportion of data on test.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    split_handler = SplitHandler(
        train_plan=args.train_plan, 
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio
    )

    split_handler.split_on_questions(dataset_name=args.dataset_name, save=args.save, debug=args.debug, verbose=args.verbose)

if __name__ == "__main__":
    main()