import argparse
from calyapo.data_preprocessing.funcs.data_combiner import split_combine

def main():
    parser = argparse.ArgumentParser(description="Takes in cleaned json data and generates train, val , tests splits in <prompt:completion> format for finetuning")
    parser.add_argument("train_plan", type=str, nargs='?', default='ideology_to_trump', help="Name of training plan to finetune on.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    split_combine(train_plan=args.train_plan, save=args.save, debug=args.debug)

if __name__ == "__main__":
    main()