import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import shap
from shap.maskers import Text

# Paths
MODEL_PATH = "new_reddit_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
TEST_DATA_PATH = "extracted_test_data.csv"

SHAP_SAMPLES_SIZE = 100
SHAP_TOP_TOKENS = 3

include_shap = input("Include SHAP Top Words Attribution? (y/n): ").strip().lower() == 'y'

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
print("\nPredictions saved to manual_test_output.csv")


if include_shap:
    # === SHAP Top Words Attribution ===
    # Define function to explain text
    def predict_fn(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=100)
        return model.predict(padded)

    # Use default whitespace masker
    masker = Text()
    explainer = shap.Explainer(predict_fn, masker)

    # Run SHAP on all test comments (or sample if large)
    sample_size = min(SHAP_SAMPLES_SIZE, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    texts = df_sample['clean_comment'].tolist()
    shap_values = explainer(texts)

    # Extract top 5 contributing tokens for predicted class
    important_tokens = []
    for i, text in enumerate(texts):
        pred_class = np.argmax(predict_fn([text])[0])
        tokens = shap_values.data[i]
        scores = shap_values.values[i]
        class_scores = scores[:, pred_class]

        # Pair tokens with scores and sort by |score| descending
        top_tokens = sorted(zip(tokens, class_scores), key=lambda x: abs(x[1]), reverse=True)
        top_words = [f"{tok}:{score:.3f}" for tok, score in top_tokens[:SHAP_TOP_TOKENS]]
        important_tokens.append(', '.join(top_words))

    # Add to DataFrame
    df_sample['important_words'] = important_tokens

    df_sample.to_csv("manual_test_shap_output.csv", index=False)
    print("\nSHAP Top Words Attribution saved to manual_test_shap_output.csv")

