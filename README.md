# Reddit Sentiment Analysis

## Repository Contents
- **Dataset**: `Reddit_Data.csv`
- **Training and Evaluation Scripts**
- **Model Prediction Tools**: `manual_test.py`
- **SHAP Explanation Tools**
- **Scripts for Generating Plots and Analysis**:
  - `plot_hyperparameter_effects.py`
  - `rank_best_models.py`

---

## Reproducing Results

### 1. Clone the Repository
```
git clone https://github.com/aonlo/redditsentimentanalysis
cd redditsentimentanalysis
```

### 2. Install Dependencies
Use a virtual environment and install the required packages (e.g., TensorFlow, pandas, matplotlib, scikit-learn, seaborn, SHAP):  
```
pip install -r requirements.txt
```

### 3. Train and Evaluate All Models
Run the main training script to conduct the full grid search:
```
python sentiment_training.py
```

This will:
- Train all **54 model configurations**
- Save each model
- Record evaluation metrics to `grid_search_results.csv`

⚠️ **Note**: This process may take several hours depending on your device’s computing power. An estimate for the remaining time is displayed in the console and improves as more models are trained.

### 4. Generate Visualizations
Once training is complete:
```
python plot_hyperparameter_effects.py
python rank_best_models.py
```

### 5. Review the Results
- **Top 10 models** will be saved in `top_10_models.csv`
- **Metric comparison plots** will appear in `plots/` and `top_model_plots/`
- Use `manual_test.py` to load any trained model and test predictions on manual input or a new CSV file.

### 6. (Optional) Interpret Predictions
If SHAP is enabled in `manual_test.py`, the script will generate SHAP visualizations for word-level attribution in sentiment predictions.
