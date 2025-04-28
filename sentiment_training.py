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

# Paths
MODEL_PATH = "sentiment_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
ENCODER_PATH = "label_encoder.pkl"
NEW_DATA_PATH = "new_reddit_data.csv"

# Defaults
default_optimizer = "adam"
default_learning_rate = 0.001
default_batch_size = 64

# User input prompts
model_path = input(f"Enter model path (default: {MODEL_PATH}): ").strip()
if not model_path:
    model_path = MODEL_PATH

train_data = input(f"Enter data path (default: {NEW_DATA_PATH}): ").strip()
if not train_data:
    train_data = NEW_DATA_PATH

optimizer_choice = input(f"Choose optimizer (adam, rmsprop, sgd) [default: {default_optimizer}]: ").strip().lower()
if not optimizer_choice:
    optimizer_choice = default_optimizer

learning_rate = input(f"Enter learning rate [default: {default_learning_rate}]: ").strip()
if not learning_rate:
    learning_rate = default_learning_rate
else:
    learning_rate = float(learning_rate)

batch_size = input(f"Enter batch size [default: {default_batch_size}]: ").strip()
if not batch_size:
    batch_size = default_batch_size
else:
    batch_size = int(batch_size)

# Create a log under logs
os.makedirs("logs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"logs/training_log_{timestamp}.txt"
# Build log content
log_content = f"""
Training Session - {timestamp}
-----------------------------------
Model Path: {model_path}
Data Path: {train_data}
Optimizer: {optimizer_choice}
Learning Rate: {learning_rate}
Batch Size: {batch_size}
-----------------------------------
"""
# Write log file
with open(log_filename, "w") as f:
    f.write(log_content)
print(f"Training configuration saved to {log_filename}")

# Map string to optimizer instance
optimizer_map = {
    "adam": Adam(learning_rate=learning_rate),
    "rmsprop": RMSprop(learning_rate=learning_rate),
    "sgd": SGD(learning_rate=learning_rate)
}
optimizer = optimizer_map.get(optimizer_choice, Adam(learning_rate=learning_rate))

# Load new data
df = pd.read_csv(train_data)
df = df[df['clean_comment'].notna() & (df['clean_comment'].str.strip() != '')]
df['clean_comment'] = df['clean_comment'].astype(str)

# First-time training if model/tokenizer/encoder not found
if not (os.path.exists(model_path) and os.path.exists(TOKENIZER_PATH) and os.path.exists(ENCODER_PATH)):
    print("No existing model or preprocessing tools found. Training from scratch...")

    # Initialize and fit tokenizer and encoder
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['clean_comment'])
    sequences = tokenizer.texts_to_sequences(df['clean_comment'])
    X = pad_sequences(sequences, maxlen=100)

    label_encoder = LabelEncoder()
    y = to_categorical(label_encoder.fit_transform(df['category']))

    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=100),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    model.fit(X, y, epochs=5, batch_size=batch_size, callbacks=[early_stop])

    # Save everything
    model.save(model_path)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print("Initial model and tools trained and saved.")

else:
    print("Found saved model. Fine-tuning with new data...")

    # Load tools
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    # Preprocess data
    sequences = tokenizer.texts_to_sequences(df['clean_comment'])
    X = pad_sequences(sequences, maxlen=100)
    y = to_categorical(label_encoder.transform(df['category']))

    # Split for evaluation
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, stratify=y)

    # Load model
    model = load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Evaluate BEFORE training 
    y_pred_before = model.predict(X_eval)
    y_classes_before = np.argmax(y_pred_before, axis=1)
    y_true_eval = np.argmax(y_eval, axis=1)
    present_labels_before = unique_labels(y_true_eval, y_classes_before)
    present_target_names_before = [str(label_encoder.classes_[i]) for i in present_labels_before]

    print("Performance BEFORE training:")
    print(classification_report(
        y_true_eval,
        y_classes_before,
        labels=present_labels_before,
        target_names=present_target_names_before
    ))

    # Fine-tune 
    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=3, batch_size=batch_size, callbacks=[early_stop])

    # Evaluate AFTER training 
    y_pred_after = model.predict(X_eval)
    y_classes_after = np.argmax(y_pred_after, axis=1)

    present_labels_after = unique_labels(y_true_eval, y_classes_after)
    present_target_names_after = [str(label_encoder.classes_[i]) for i in present_labels_after]

    print("Performance AFTER training:")
    print(classification_report(
        y_true_eval,
        y_classes_after,
        labels=present_labels_after,
        target_names=present_target_names_after
    ))

    # Save model
    model.save(model_path)
    print("Model fine-tuned and saved.")
