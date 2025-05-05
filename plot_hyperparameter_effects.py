import os
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("grid_search_results.csv")

# Define hyperparameters and all result metrics
hyperparams = ['Optimizer', 'Learning Rate', 'Batch Size', 'Epochs']
metrics = [
    'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score',
    'Train Loss', 'Validation Loss',
    'Validation Precision', 'Validation Recall', 'Validation F1 Score',
    'Avg Epoch Time (s)'
]

# Create output folder
os.makedirs("plots", exist_ok=True)

# For each metric and hyperparameter, compute group means and plot clean bars
for metric in metrics:
    for param in hyperparams:
        plt.figure(figsize=(10, 6))
        grouped = df.groupby(param)[metric]
        means = grouped.mean()

        bars = plt.barh(means.index.astype(str), means.values, color='skyblue', edgecolor='black')
        plt.xlabel(metric)
        plt.title(f"{metric} by {param}")
        plt.tight_layout()

        # Label each bar
        for bar, val in zip(bars, means.values):
            plt.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va='center')

        filename = f"plots/{metric.replace(' ', '_').lower()}_by_{param.replace(' ', '_').lower()}.png"
        plt.savefig(filename)
        plt.close()

print("✅ Clean comparison plots saved to /plots")
