import pandas as pd
import subprocess

# Create new Reddit data 
# Possibly from user input
data = pd.read_csv("Reddit_Data.csv")

# Save to CSV
new_df = pd.DataFrame(data)
new_df.to_csv("new_reddit_data.csv", index=False)

# Trigger training script
subprocess.run(["python", "sentiment_training.py"])
