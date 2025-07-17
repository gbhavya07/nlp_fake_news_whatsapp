import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

# Load the dataset
cleaned_file_path = 'new_dataset_indian.csv'
news_data = pd.read_csv(cleaned_file_path)

# Define features and labels
X = news_data['cleaned_text']  # Preprocessed text
y = news_data['label']         # Labels (FAKE or REAL)

# Encode labels (FAKE -> 0, REAL -> 1)
y = y.map({"FAKE": 0, "REAL": 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)  # Reduce feature space further
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# Define individual models
nb_model = MultinomialNB()  # Naive Bayes
logreg = LogisticRegression(max_iter=500, class_weight="balanced", C=0.1)  # Logistic Regression

# Combine models into a Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ("nb", nb_model),
    ("logreg", logreg)
], voting="soft")  # Use soft voting for probability-based decisions

# Train the ensemble model on the resampled training set
print("Training the Voting Classifier...")
voting_clf.fit(X_train_resampled, y_train_resampled)
print("Model training complete.")

# Evaluate the model on the test set
y_test_pred = voting_clf.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))

# Adjusting prediction threshold for fake news
import pickle

# Save the trained VotingClassifier
with open('voting_classifier.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)

# Save the trained TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
