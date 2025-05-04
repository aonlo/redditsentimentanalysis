import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load grid search results
df = pd.read_csv("grid_search_results.csv")

# Metrics setup
positive_metrics = [
    'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score',
    'Validation Precision', 'Validation Recall', 'Validation F1 Score'
]
negative_metrics = ['Train Loss', 'Validation Loss', 'Avg Epoch Time (s)']
all_metrics = positive_metrics + negative_metrics

# Normalize for scoring
df_norm = df.copy()
scaler = MinMaxScaler()
df_norm[positive_metrics] = scaler.fit_transform(df[positive_metrics])
df_norm[negative_metrics] = scaler.fit_transform(df[negative_metrics])
df_norm[negative_metrics] = 1.0 - df_norm[negative_metrics]  # invert lower-is-better

# ✅ Custom Weighted Combined Score
df_norm['Combined Score'] = (
    df_norm['Test F1 Score'] * 0.30 +
    df_norm['Validation F1 Score'] * 0.30 +
    df_norm['Validation Loss'] * 0.15 +   # inverted
    df_norm['Avg Epoch Time (s)'] * 0.10 +  # inverted
    df_norm['Test Accuracy'] * 0.10 +
    df_norm['Validation Recall'] * 0.05
)

# Get top 10 ranked models
top_10_norm = df_norm.sort_values('Combined Score', ascending=False).head(10)
top_10 = df.loc[top_10_norm.index].copy()
top_10['Combined Score'] = top_10_norm['Combined Score'].values
top_10.reset_index(drop=True, inplace=True)

# Save ranked results
top_10.to_csv("top_10_models.csv", index=False)
os.makedirs("top_model_plots", exist_ok=True)

# 📊 Create horizontal comparison plots for each metric
for metric in all_metrics + ['Combined Score']:
    sorted_df = top_10.sort_values(by=metric, ascending=False)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_df["Model File"], sorted_df[metric], color='skyblue', edgecolor='black')
    plt.title(f"{metric} Across Top 10 Models")
    plt.xlabel(metric)
    plt.gca().invert_yaxis()

    # Label each bar with the actual value
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{width:.4f}", va='center')

    plt.tight_layout()
    filename = f"top_model_plots/comparison_{metric.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    plt.close()

print("✅ Top 10 models saved to top_10_models.csv")
print("📊 Horizontal comparison plots saved in /top_model_plots/")
