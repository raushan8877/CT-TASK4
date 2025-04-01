# CODETECH-TASK4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
try:
    data = pd.read_csv("spam.csv", encoding='latin-1')
    data = data[['v1', 'v2']]  # Selecting only necessary columns
    data.columns = ['label', 'message']
except FileNotFoundError:
    print("Error: The dataset file 'spam.csv' was not found. Please upload the file and try again.")
    data = None
except Exception as e:
    print(f"Error loading dataset: {e}")
    data = None

if data is not None:
    # Encode labels
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Check for missing values
    data.dropna(inplace=True)

    # Splitting dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

    # Creating a text processing and classification pipeline
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultinomialNB())
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Visualizing Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
else:
    print("Program terminated due to missing dataset.")
