import argparse
from calyapo.data_preprocessing.raw_handler import RawHandler

def main():
    parser = argparse.ArgumentParser(description="Preliminary cleaning of IGS data")
    parser.add_argument("--dataset_name", type=str, nargs='?', default='IGS', help="Name of dataset to clean.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    raw_handler = RawHandler()
    raw_handler.clean_dataset(args.dataset_name, save=args.save, debug=args.debug, verbose=args.verbose)

if __name__ == "__main__":
    main()