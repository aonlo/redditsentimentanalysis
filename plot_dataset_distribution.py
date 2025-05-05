import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
DATA_PATH = "Reddit_Data.csv"  # Change this to your dataset path
df = pd.read_csv(DATA_PATH)

# Drop empty comments if any
df = df[df['clean_comment'].notna() & (df['clean_comment'].str.strip() != '')]

# Plot category distribution
plt.figure(figsize=(8, 5))
df['category'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Distribution of Sentiment Categories")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("dataset_distribution.png")
plt.show()
