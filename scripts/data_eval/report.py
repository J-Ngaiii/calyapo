import argparse
import sys
import json
from pathlib import Path
from calyapo.data_eval.reporter import Reporter 

def main():
    parser = argparse.ArgumentParser(description="Generates performance and accuracy reports from tabularized data.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school', help="Name of training plan.")
    parser.add_argument("--run_keyword", type=str, nargs='?', default='aurora', help="Keyword for the report folder.")
    parser.add_argument("--root_path", type=str, nargs='?', default='.', help="Folder root from which to run analysis.")
    parser.add_argument("--acc_report", action=argparse.BooleanOptionalAction, default=True, help="Run Reporter class' accuracy analysis.")
    parser.add_argument("--show_plots", action=argparse.BooleanOptionalAction, default=False, help="Display plots during execution.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rep = Reporter(
        train_plan=args.train_plan,
        run_keyword=args.run_keyword,
        root_path=args.root_path,
        debug=args.debug,
        verbose=args.verbose
    )

    if args.verbose:
        print(f"--- Starting Reporting Pipeline ---")
        print(f"Train Plan: {args.train_plan}")
        print(f"Keyword:    {args.run_keyword}")

    if args.acc_report:
        rep.accuracy(show_plots=args.show_plots)

    if args.verbose:
        print(f"--- Reporting Complete. Results saved in: {rep.results_folder} ---")

if __name__ == "__main__":
    main()