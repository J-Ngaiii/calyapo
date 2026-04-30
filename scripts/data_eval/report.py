import argparse
import sys
import json
from pathlib import Path
from calyapo.data_eval.reporter import Reporter 

def main():
    parser = argparse.ArgumentParser(description="Runs the full analysis pipeline.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school')
    parser.add_argument("--run_keyword", type=str, nargs='?', default='aurora')
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rep = Reporter(
        train_plan=args.train_plan,
        run_keyword=args.run_keyword,
        verbose=args.verbose,
        debug=args.debug
    )

    # 1. Basic Accuracy Plot
    rep.accuracy()

    # 2. Generate Demographic Crosstabs (Required for KL Analysis)
    rep.generate_crosstabs()

    # 3. Calculate Distributional metrics (KL/Wasserstein)
    rep.distributional_accuracy(demog_col_indices=[0])
if __name__ == "__main__":
    main()