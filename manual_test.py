import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
MODEL_PATH = "sentiment_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
TEST_DATA_PATH = "test_reddit_data.csv"

# Load model and tools
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Load and clean test data
df = pd.read_csv(TEST_DATA_PATH)
df = df[df['clean_comment'].notna() & (df['clean_comment'].str.strip() != '')]
df['clean_comment'] = df['clean_comment'].astype(str)

# Preprocess 
sequences = tokenizer.texts_to_sequences(df['clean_comment'])
X = pad_sequences(sequences, maxlen=100)

# Predict 
pred_probs = model.predict(X)
pred_classes = np.argmax(pred_probs, axis=1)
pred_labels = label_encoder.inverse_transform(pred_classes)

# Add predictions to DataFrame 
df['predicted_category'] = pred_labels

# Visualize category distribution 
plt.figure(figsize=(6, 4))
df['predicted_category'].value_counts().plot(kind='bar')
plt.title("Predicted Category Distribution")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Save predictions to CSV 
df.to_csv("predictions_output.csv", index=False)
print("\nPredictions saved to predictions_output.csv")
