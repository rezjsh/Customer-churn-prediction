# Customer Churn Prediction

## Project Overview

This repository delivers a complete customer churn prediction workflow for a subscription-based business. The project covers data preparation, feature engineering, model training, evaluation, and inference.

## Key Objectives

- Predict whether a customer is likely to churn.
- Demonstrate a repeatable Python machine learning pipeline.
- Track model performance with explainable metrics.
- Provide clear instructions for local setup and usage.

## Dataset

The dataset should include customer account and usage details such as:

- Customer identifier
- Account tenure
- Subscription plan or contract type
- Monthly charges and total charges
- Service usage features
- Churn label (target variable)

> Note: Replace these placeholders with the exact file names and schema used in this repository.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Customer-churn-prediction.git
   cd Customer-churn-prediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   source venv/bin/activate  # macOS / Linux
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Exploratory analysis

Open the exploratory notebook or run the analysis script to inspect the data:

```bash
jupyter notebook
```

Then open the relevant notebook file in `notebooks/`.

### Training the model

Run the training pipeline:

```bash
python src/train.py
```

### Evaluating the model

Run the evaluation pipeline:

```bash
python src/evaluate.py
```
```

### Generating predictions

Use the inference script for new customer data:

```bash
python src/predict.py --input data/new_customers.csv --output predictions.csv
```

## Project Structure

- `data/` - raw and processed dataset files
- `notebooks/` - exploratory data analysis and modeling notebooks
- `src/` - training, evaluation, and inference scripts
- `models/` - trained model artifacts and preprocessing objects
- `reports/` - evaluation summaries and charts
- `requirements.txt` - Python dependencies
- `README.md` - project documentation

> Adjust these paths to match your repository layout.

## Model Pipeline Overview

1. Data ingestion and cleaning
2. Feature engineering
3. Train/test splitting
4. Model training with cross-validation
5. Performance evaluation using classification metrics
6. Exporting the model and preprocessing pipeline

## Evaluation Metrics

Common metrics for churn prediction include:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- Confusion matrix

## Results

Summarize the best model and key results here, for example:

- Best algorithm used: `RandomForestClassifier`, `XGBoost`, or `LogisticRegression`
- Best validation AUC: `0.87`
- Most important features: `tenure`, `monthly_charges`, `contract_type`, `support_calls`

## Customization

To adapt this repository to your dataset:

1. Replace dataset files in `data/`.
2. Update preprocessing and feature engineering in `src/preprocess.py` or notebook cells.
3. Adjust model hyperparameters in `src/train.py`.
4. Rerun training and evaluation.

## Contributing

Contributions are welcome. Please fork the repository, create a branch, and submit a pull request.

## License

This project is released under the MIT License. Update this section if a different license applies.
 Customer-churn-prediction