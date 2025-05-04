import os
import datetime
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import csv
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

# Define grid search hyperparameters
optimizers = ["adam", "sgd"]
learning_rates = [0.002, 0.001, 0.0005]
batch_sizes = [32, 64, 128]
epochs_list = [7, 5, 3]
patience = 2

# Paths
MODEL_PATH = "sentiment_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
NEW_DATA_PATH = "Reddit_Data.csv"

# Load new data
df = pd.read_csv(NEW_DATA_PATH)
df = df[df['clean_comment'].notna() & (df['clean_comment'].str.strip() != '')]
df['clean_comment'] = df['clean_comment'].astype(str)

# Preprocess: tokenize and encode once
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_comment'])
sequences = tokenizer.texts_to_sequences(df['clean_comment'])
X = pad_sequences(sequences, maxlen=100)
label_encoder = LabelEncoder()
y_labels = label_encoder.fit_transform(df['category'])
y = to_categorical(y_labels)
# First split: Train + Temp (which will become val + test)
X_train, X_temp, y_train, y_temp, y_labels_train, y_labels_temp = train_test_split(
    X, y, y_labels, test_size=0.3, stratify=y_labels, random_state=42)

# Second split: Validation and Test (from Temp)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_labels_temp, random_state=42)

# Save tokenizer and encoder
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

# Prepare CSV log
csv_path = "grid_search_results.csv"
with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Model File", "Optimizer", "Learning Rate", "Batch Size", "Epochs",
        "Test Accuracy",
        "Train Loss", "Validation Loss",
        "Validation Precision", "Validation Recall", "Validation F1 Score",
        "Test Precision", "Test Recall", "Test F1 Score",
        "Avg Epoch Time (s)"
    ])

    # Grid search loop
    from itertools import product
    grid = list(product(optimizers, learning_rates, batch_sizes, epochs_list))
    for opt_name, lr, batch_size, epochs in tqdm(grid, desc="Grid Search", unit="combo"):
        print(f"Training: optimizer={opt_name}, lr={lr}, batch={batch_size}, epochs={epochs}")

        # Select optimizer
        if opt_name == "adam":
            optimizer = Adam(learning_rate=lr)
        elif opt_name == "sgd":
            optimizer = SGD(learning_rate=lr)

        # Build model
        vocab_size = len(tokenizer.word_index) + 1
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=128, input_length=100),
            Bidirectional(LSTM(64)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        elapsed_time = time.time() - start_time
        avg_epoch_time = elapsed_time / len(history.history['loss'])  # avg per epoch

        # Predict on validation set
        y_pred_probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_val, axis=1)

        # Compute metrics
        val_accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        # Ensure models/ directory exists
        os.makedirs("models", exist_ok=True)

        # Save the model with a unique numeric name
        model_index = len(os.listdir("models"))
        model_path = f"models/model_{model_index}.keras"
        model.save(model_path)

        # Evaluate on test set
        y_test_probs = model.predict(X_test, verbose=0)
        y_test_pred = np.argmax(y_test_probs, axis=1)
        y_test_true = np.argmax(y_test, axis=1)

        test_accuracy = accuracy_score(y_test_true, y_test_pred)
        test_precision = precision_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test_true, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test_true, y_test_pred, average='weighted', zero_division=0)

        writer.writerow([
            model_path, opt_name, lr, batch_size, epochs,
            round(test_accuracy, 4),
            round(train_loss, 4), round(val_loss, 4),
            round(precision, 4), round(recall, 4), round(f1, 4),
            round(test_precision, 4), round(test_recall, 4), round(test_f1, 4),
            round(avg_epoch_time, 4)
        ])

print(f"\nGrid search completed. Results saved to: {csv_path}")