import os
import pandas as pd
import seaborn as sns
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

# Ensure output directory exists
os.makedirs("plots", exist_ok=True)

# Generate one plot per (hyperparameter, metric) pair
for param in hyperparams:
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=df, x=param, y=metric)
        plt.title(f"{metric} by {param}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"plots/{metric.replace(' ', '_').lower()}_by_{param.replace(' ', '_').lower()}.png"
        plt.savefig(filename)
        plt.close()

print("✅ Plots saved in the /plots folder.")
