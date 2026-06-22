# Customer Churn Prediction

A machine learning project that predicts whether a bank customer is likely to churn (exit the bank), using an Artificial Neural Network (ANN) built with TensorFlow/Keras and served through an interactive Streamlit web app.

## Overview

The model is trained on the [Churn_Modelling.csv](Churn_Modelling.csv) dataset and predicts customer churn based on attributes such as credit score, geography, gender, age, tenure, balance, number of products, credit card ownership, account activity, and estimated salary.

## Project Structure

```
.
├── app.py                      # Streamlit web app for live predictions
├── experiments.ipynb           # Data preprocessing, EDA, and model training notebook
├── prediction.ipynb            # Notebook for loading the saved model and running predictions
├── Churn_Modelling.csv          # Raw dataset
├── model.h5                     # Trained Keras ANN model
├── scaler.pkl                   # Fitted StandardScaler for input features
├── Label_Encoder_gender.pkl     # Fitted LabelEncoder for the Gender column
├── onehot_encoder_geo.pkl       # Fitted OneHotEncoder for the Geography column
├── logs/                        # TensorBoard training logs
└── requirements.txt             # Python dependencies
```

## Model

A simple feed-forward ANN built with Keras:

- **Input layer** → Dense(64, activation="relu")
- **Hidden layer** → Dense(32, activation="relu")
- **Output layer** → Dense(1, activation="sigmoid")

Trained with the Adam optimizer (`lr=0.01`), binary cross-entropy loss, `EarlyStopping` (patience=10, restores best weights), and `TensorBoard` logging, for up to 100 epochs.

## Data Preprocessing

1. Drop irrelevant columns: `RowNumber`, `CustomerId`, `Surname`.
2. Encode `Gender` with `LabelEncoder`.
3. One-hot encode `Geography` with `OneHotEncoder`.
4. Split into train/test sets (80/20).
5. Scale features with `StandardScaler`.

The fitted encoders and scaler are saved as `.pkl` files so the exact same transformations can be reapplied at inference time.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Streamlit app

```bash
streamlit run app.py
```

This opens a browser UI where you can input customer details (geography, gender, age, balance, credit score, tenure, number of products, etc.) and get a churn prediction with probability.

### Retrain the model

Open and run [experiments.ipynb](experiments.ipynb) to reproduce preprocessing, training, and artifact generation (`model.h5`, `scaler.pkl`, encoder `.pkl` files).

### Run predictions in a notebook

Open [prediction.ipynb](prediction.ipynb) to load the saved model/encoders and generate predictions programmatically.

### View training metrics

```bash
tensorboard --logdir logs/fit
```

## Dependencies

- tensorflow==2.15.0
- pandas
- numpy
- scikit-learn
- tensorboard
- matplotlib
- streamlit
