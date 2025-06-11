#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
import nltk

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# === Preprocessing ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(w) for w in words])

# === Train XGBoost Classifier ===
def train_xgboost_model(train_file):
    df = pd.read_excel(train_file)

    df['product_text'] = df[['product_desc_1', 'product_desc_2', 'central_description']] \
        .fillna('').astype(str).agg(' '.join, axis=1).apply(preprocess_text)

    df['full_category'] = df[['FTICategory1', 'FTICategory2', 'FTICategory3']] \
        .fillna('').astype(str).agg(' > '.join, axis=1)

    print("✅ Loaded and preprocessed training data.")

    # Encode text
    model = SentenceTransformer('all-mpnet-base-v2')
    X = model.encode(df['product_text'].tolist(), show_progress_bar=True)

    # Encode target labels
    y_text = df['full_category']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_text)

    # Train XGBoost
    clf = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500
    )
    clf.fit(X, y_encoded)

    # === Training evaluation ===
    y_train_pred_encoded = clf.predict(X)
    y_train_pred = label_encoder.inverse_transform(y_train_pred_encoded)

    df['Predicted_full_category'] = y_train_pred
    split_cols = df['Predicted_full_category'].str.split(' > ', expand=True)
    while split_cols.shape[1] < 3:
        split_cols[split_cols.shape[1]] = None
    split_cols.columns = ['Pred_CATEGORY1', 'Pred_CATEGORY2', 'Pred_CATEGORY3']
    df = pd.concat([df, split_cols], axis=1)

    print("\n=== Training Set Evaluation ===")
    for level in ['1', '2', '3']:
        true_col = f'FTICategory{level}'
        pred_col = f'Pred_CATEGORY{level}'
        acc = accuracy_score(df[true_col], df[pred_col])
        f1 = classification_report(df[true_col], df[pred_col], output_dict=True)['weighted avg']['f1-score']
        print(f"CATEGORY{level} Accuracy: {acc:.4f}, F1: {f1:.4f}")

    print("✅ Trained XGBoost model.")
    return clf, model, label_encoder

# === Predict and Evaluate ===
def predict_and_evaluate(clf, model, label_encoder, test_file, output_file):
    df = pd.read_excel(test_file)
    df['product_text'] = df[['product_desc_1', 'product_desc_2', 'central_description']] \
        .fillna('').astype(str).agg(' '.join, axis=1).apply(preprocess_text)

    X_test = model.encode(df['product_text'].tolist(), show_progress_bar=True)
    y_pred_encoded = clf.predict(X_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    df['Predicted_full_category'] = y_pred_labels.astype(str)

    # Safe split with fallback
    split_cols = df['Predicted_full_category'].str.split(' > ', expand=True)
    while split_cols.shape[1] < 3:
        split_cols[split_cols.shape[1]] = None
    split_cols.columns = ['Pred_CATEGORY1', 'Pred_CATEGORY2', 'Pred_CATEGORY3']
    df = pd.concat([df, split_cols], axis=1)

    # Evaluate
    print("\n=== Test Set Evaluation ===")
    for level in ['1', '2', '3']:
        true_col = f'FTICategory{level}'
        pred_col = f'Pred_CATEGORY{level}'
        acc = accuracy_score(df[true_col], df[pred_col])
        f1 = classification_report(df[true_col], df[pred_col], output_dict=True)['weighted avg']['f1-score']
        print(f"CATEGORY{level} Accuracy: {acc:.4f}, F1: {f1:.4f}")

    df.to_excel(output_file, index=False)
    print(f"\n✅ Saved predictions to {output_file}")


# In[ ]:


clf, model, label_encoder = train_xgboost_model("training_set.xlsx")
predict_and_evaluate(clf, model, label_encoder, "Test_set.xlsx", "Test_Final_with_predictions.xlsx")

