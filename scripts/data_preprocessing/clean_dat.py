import argparse
from calyapo.data_preprocessing.clean_datasets import build_steering_dataset

def main():
    parser = argparse.ArgumentParser(description="Build and clean raw CSVs into JSON files.")
    parser.add_argument("dataset_name", type=str, nargs='?', default='IGS', help="Name of dataset to clean.")
    parser.add_argument("train_plan", type=str, nargs='?', default='ideology_to_trump', help="Name of training plan to finetune on.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    build_steering_dataset(
        dataset_name=args.dataset_name, 
        train_plan=args.train_plan, 
        save=args.save, 
        debug=args.debug
    )

if __name__ == "__main__":
    main()