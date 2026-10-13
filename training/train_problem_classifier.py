import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "financial_problems.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_problem_classifier():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)

    print(f"Loaded {len(df)} financial problems across {df['problemType'].nunique()} classes.")
    X = df["text"].str.lower().str.strip()
    y = df["problemType"]

    class_counts = y.value_counts()
    can_stratify = (class_counts >= 2).all()
    stratify = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    classifier = LogisticRegression(max_iter=1000, C=3.0, random_state=42)
    classifier.fit(X_train_vec, y_train)

    y_pred = classifier.predict(X_test_vec)
    print("\n--- Problem Classifier Evaluation ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    labels = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    # Refit on full dataset for production accuracy
    X_full_vec = vectorizer.fit_transform(X)
    classifier.fit(X_full_vec, y)

    vectorizer_path = os.path.join(MODELS_DIR, "problem_vectorizer.joblib")
    classifier_path = os.path.join(MODELS_DIR, "problem_classifier.joblib")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)

    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved classifier to: {classifier_path}")

    return {
        "modelName": "Financial Problem Classifier",
        "algorithm": "TF-IDF + Logistic Regression",
        "datasetSize": len(df),
        "classesCount": int(df["problemType"].nunique()),
        "classes": labels,
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1Score": round(float(f1), 4),
        "confusionMatrix": cm,
    }


if __name__ == "__main__":
    train_problem_classifier()
