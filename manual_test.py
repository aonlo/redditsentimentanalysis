import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

# Paths
MODEL_PATH = "new_reddit_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
TEST_DATA_PATH = "extracted_test_data.csv"

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

# Preprocess text
sequences = tokenizer.texts_to_sequences(df['clean_comment'])
X = pad_sequences(sequences, maxlen=100)

# Predict
pred_probs = model.predict(X)
pred_classes = np.argmax(pred_probs, axis=1)
pred_labels = label_encoder.inverse_transform(pred_classes)

# Add predictions to DataFrame
df['predicted_category'] = pred_labels

# Accuracy calculation
if 'category' in df.columns:
    true_classes = label_encoder.transform(df['category'])
    accuracy = accuracy_score(true_classes, pred_classes)
    print(f"Accuracy: {accuracy:.4f}")

    # Precision, Recall, F1-score
    try:
        target_names = [str(label) for label in label_encoder.classes_]
        report = classification_report(true_classes, pred_classes, target_names=target_names)
    except:
        report = classification_report(true_classes, pred_classes)

    print("\nClassification Report:\n")
    print(report)
else:
    print("Warning: 'category' column not found in test data. Accuracy cannot be computed.")

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
df.to_csv("manual_test_output.csv", index=False)
print("\nPredictions saved to predictions_output.csv")
