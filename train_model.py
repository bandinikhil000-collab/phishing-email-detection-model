"""
PhishGuard AI — Model Training & Evaluation Pipeline
Trains a TF-IDF + Logistic Regression machine learning model on labeled email datasets.
Saves the trained vectorizer, model, and detailed evaluation metrics.
"""

import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "phishing_emails_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "phishing_classifier.joblib")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")


def clean_text(text: str) -> str:
    """
    Standardize text: lowercases, removes abnormal whitespace,
    and replaces URLs, emails, and currencies with informative tokens.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Replace URLs
    text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)
    # Replace email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " <EMAIL> ", text)
    # Replace monetary values
    text = re.sub(r"[\$£€]\s*\d+([,\.]\d+)?", " <CURRENCY> ", text)
    # Replace excessive punctuation
    text = re.sub(r"[!]{2,}", " !! ", text)
    text = re.sub(r"[\?]{2,}", " ?? ", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_and_evaluate():
    print("=" * 60)
    print("PhishGuard AI — Training Machine Learning Model")
    print("=" * 60)

    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Generating default dataset first...")
        from data.dataset_builder import create_dataset_csv
        create_dataset_csv()

    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"Total samples loaded: {len(df)}")
    print(f"Class distribution: Legitimate (0) = {(df['label'] == 0).sum()}, Phishing (1) = {(df['label'] == 1).sum()}")

    # Preprocess text
    print("\n[1/5] Preprocessing email texts...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Train / Test split (80% train, 20% test, stratified)
    print("\n[2/5] Splitting dataset (80% train / 20% test, stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'],
        test_size=0.20,
        random_state=42,
        stratify=df['label']
    )
    print(f"Training samples: {len(X_train_raw)} | Testing samples: {len(X_test_raw)}")

    # TF-IDF Vectorization
    print("\n[3/5] Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english',
        min_df=1
    )
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())} features")

    # Model Training: Logistic Regression with balanced weights
    print("\n[4/5] Training Logistic Regression Classifier...")
    model = LogisticRegression(
        C=1.5,
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        solver='liblinear'
    )
    model.fit(X_train, y_train)

    # Comparison test: Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_preds)

    # Evaluation
    print("\n[5/5] Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    # cm: [[TN, FP], [FN, TP]]

    print("\n--- Test Set Evaluation Metrics ---")
    print(f"Accuracy:          {acc * 100:.2f}%")
    print(f"Precision:         {prec * 100:.2f}%")
    print(f"Recall:            {rec * 100:.2f}%")
    print(f"F1-Score:          {f1 * 100:.2f}%")
    print(f"Naive Bayes Acc:   {nb_acc * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  [TN: {cm[0][0]}, FP: {cm[0][1]}]")
    print(f"  [FN: {cm[1][0]}, TP: {cm[1][1]}]")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Phishing (1)"]))

    # Extract Top Informative Features (Coefficients)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_phish_idx = np.argsort(coefs)[-12:][::-1]
    top_legit_idx = np.argsort(coefs)[:12]

    top_phishing_words = [
        {"feature": feature_names[i], "weight": round(float(coefs[i]), 3)}
        for i in top_phish_idx
    ]
    top_legit_words = [
        {"feature": feature_names[i], "weight": round(float(abs(coefs[i])), 3)}
        for i in top_legit_idx
    ]

    # Save artifacts
    print(f"\nSaving classifier to: {MODEL_FILE}")
    joblib.dump(model, MODEL_FILE)

    print(f"Saving vectorizer to: {VECTORIZER_FILE}")
    joblib.dump(vectorizer, VECTORIZER_FILE)

    metrics_payload = {
        "model_type": "Logistic Regression with TF-IDF Vectorization",
        "comparison_model": f"Multinomial Naive Bayes (Accuracy: {nb_acc*100:.1f}%)",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(df),
        "train_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "vocabulary_size": len(feature_names),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": {
                "true_negative": cm[0][0],
                "false_positive": cm[0][1],
                "false_negative": cm[1][0],
                "true_positive": cm[1][1]
            }
        },
        "top_phishing_indicators": top_phishing_words,
        "top_legitimate_indicators": top_legit_words,
        "hyperparameters": {
            "C": 1.5,
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_features": 2500,
            "ngram_range": [1, 2]
        }
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved evaluation metrics to: {METRICS_FILE}")
    print("=" * 60)
    print("Model Training & Evaluation Completed Successfully!")
    print("=" * 60)
    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()
