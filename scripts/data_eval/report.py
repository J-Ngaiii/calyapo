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
    parser.add_argument("--only_acc", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only_crosstab", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only_conf", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only_pred", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only_distrib", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only_dist_align", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only_metric_consistency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--full_analysis", action=argparse.BooleanOptionalAction, default=False)
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


    if args.full_analysis:
        rep.accuracy()
        rep.generate_crosstabs()
        rep.confidence_analysis()
        rep.distributional_accuracy(demog_col_indices=[0])
       
        for split in ['train', 'val', 'test']:
            rep.generate_geographic_reports(split=split, train_setting=TRAIN_SETTING)
            rep.prediction_distribution_analysis(split=args.split)


        return # exit after
   
    if args.only_acc:
        rep.accuracy()
    if args.only_crosstab:
        rep.generate_crosstabs()
    if args.only_distrib:
        rep.distributional_accuracy(demog_col_indices=[0])
    if args.only_geo:
        rep.generate_geographic_reports(split=args.split, train_setting=TRAIN_SETTING)
    if args.only_conf:
        print(f"Only Confidence Analysis")
        rep.confidence_analysis(split=args.split)
    if args.only_pred:
        rep.prediction_distribution_analysis(split=args.split, granular=True)
        rep.prediction_distribution_analysis(split=args.split, granular=False)
    if args.only_dist_align:
        # rep.distributional_accuracy(demog_col_indices=[0]) # only uncomment if distrib_acc not found

        alignment_metrics = ['KL_Weighted', 'WD_Weighted']
        for metric in alignment_metrics:
            rep.distributional_alignment_analysis(split=args.split, score=metric)
    if args.only_metric_consistency:
        rep.metric_agreement_analysis(split=args.split)


if __name__ == "__main__":
    main()

