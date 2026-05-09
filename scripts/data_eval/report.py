import argparse
import sys
import json
from pathlib import Path
from calyapo.data_eval.reporter import Reporter 

def main():
    parser = argparse.ArgumentParser(description="Runs the full analysis pipeline.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school')
    parser.add_argument("--split", type=str, nargs='?', default='test')
    parser.add_argument("--run_keyword", type=str, nargs='?', default='archon')
    parser.add_argument("--only_geo", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.train_plan == 'opinion_school':
        TRAIN_SETTING = 1 
    elif args.train_plan == 'presidents_to_abortion':
        TRAIN_SETTING = 2
    else:
        raise ValueError(f"Unkown train_plan inputted: '{args.train_plan}'")
    
    rep = Reporter(
        train_plan=args.train_plan,
        run_keyword=args.run_keyword,
        verbose=args.verbose,
        debug=args.debug
    )

    if not args.only_geo:
        rep.accuracy()
        rep.generate_crosstabs()
        rep.distributional_accuracy(demog_col_indices=[0])
        rep.generate_geographic_reports(split=args.split, train_setting=TRAIN_SETTING)
    else:
        rep.generate_geographic_reports(split=args.split, train_setting=TRAIN_SETTING)
if __name__ == "__main__":
    main()