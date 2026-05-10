import pandas as pd
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

def generate_best_plot(df, split, train_plan, score: str):
    """
    We average across survey questions in the train plan
    Select best model performance by selecting the model that has min score
    """
    df = df.copy()[df['Split'] == split]
    avg_scores = df.groupby(['Demographic', 'Model'])[score].mean().reset_index()
    best_models = avg_scores.loc[avg_scores.groupby('Demographic')[score].idxmin()]
    best_models = best_models.sort_values(by=score, ascending=False)

    unique_models = best_models['Model'].unique()
    colors_palette = plt.cm.get_cmap('tab10', len(unique_models))
    model_color_map = {model: colors_palette(i) for i, model in enumerate(unique_models)}
    
    bar_colors = [model_color_map[model] for model in best_models['Model']]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(best_models['Demographic'], best_models[score], color=bar_colors, edgecolor='black')

    for bar, model_name in zip(bars, best_models['Model']):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{model_name}', 
                va='bottom', ha='center', fontsize=6, fontweight='bold')

    plt.title(f'Best Model per Demographic Group ({train_plan})', fontsize=14)
    plt.xlabel('Demographic Group', fontsize=12)
    plt.ylabel(f'Average {score} Score', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    quick_dir_name = f"{train_plan}_quick_results"
    Path(quick_dir_name).mkdir(parents=True, exist_ok=True)
    save_path = f"{quick_dir_name}/{train_plan}_{score}"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs analysis of offline inference data.") 
    parser.add_argument("--train_plan", type=str, nargs='?', default='opinion_school', help="Name of training plan.")
    parser.add_argument("--run_keyword", type=str, nargs='?', default='archon', help="Keyword corresponding with inference run.")
    parser.add_argument("--score", type=str, choices=['KL_Unweighted', 'WD_Unweighted', 'KL_Weighted', 'WD_Weighted'], default='KL_Weighted')
    
    args = parser.parse_args()
    
    df = pd.read_csv(f'inference_outputs/{args.train_plan}/reports_{args.run_keyword}/results/distributional_accuracy/summary_demog_metrics.csv')
    generate_best_plot(df=df, split='test', train_plan=args.train_plan, score=args.score)