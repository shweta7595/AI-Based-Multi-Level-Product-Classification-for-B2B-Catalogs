#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# === Preprocessing Functions ===
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    words = text.split()
    lemmatized = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(lemmatized)

def augment_text(text, factor=0.1):
    words = text.split()
    n_aug = max(1, int(len(words) * factor))
    for _ in range(n_aug):
        idx = np.random.randint(0, len(words))
        words[idx] = words[idx]
    return ' '.join(words)

# === Load and Prepare Data ===
def load_data(taxonomy_path, category_path):
    taxonomy_df = pd.read_excel(taxonomy_path)
    category_df = pd.read_csv(category_path)

    taxonomy_df.dropna(subset=['FTICategory1', 'FTICategory2', 'FTICategory3',
                               'product_desc_1', 'product_desc_2', 'central_description'], inplace=True)

    category_df.dropna(subset=['CATEGORY1', 'CATEGORY2', 'CATEGORY3'], inplace=True)

    taxonomy_df['full_category'] = taxonomy_df.apply(
        lambda row: f"{row['FTICategory1']} > {row['FTICategory2']} > {row['FTICategory3']}", axis=1)

    category_counts = taxonomy_df['full_category'].value_counts()
    valid_categories = category_counts[category_counts >= 2].index.tolist()
    taxonomy_df = taxonomy_df[taxonomy_df['full_category'].isin(valid_categories)].reset_index(drop=True)

    taxonomy_df['product_text'] = taxonomy_df[['product_desc_1', 'product_desc_2', 'central_description']].fillna('').astype(str).agg(' '.join, axis=1)
    taxonomy_df['product_text'] = taxonomy_df['product_text'].apply(preprocess_text)
    taxonomy_df['augmented_text'] = taxonomy_df['product_text'].apply(augment_text)

    df_final = pd.concat([
        taxonomy_df[['product_text', 'full_category', 'FTICategory1', 'FTICategory2', 'FTICategory3']],
        taxonomy_df[['augmented_text', 'full_category', 'FTICategory1', 'FTICategory2', 'FTICategory3']].rename(columns={'augmented_text': 'product_text'})
    ])
    return shuffle(df_final, random_state=42).reset_index(drop=True)

# === Encode Text ===
def embed_text(df, model):
    return model.encode(df['product_text'].tolist(), show_progress_bar=True)

# === Train Classifier ===
def train_classifier(X_train, y_train):
    print("Training class distribution (before SMOTE):")
    class_counts = Counter(y_train)
    print(class_counts)

    min_class_size = min(class_counts.values())
    k_neighbors = max(1, min(min_class_size - 1, 5))  # ensure at least 1

    print(f"Using SMOTE with k_neighbors={k_neighbors}")
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    clf = LogisticRegression(max_iter=5000, verbose=1, class_weight='balanced')
    clf.fit(X_bal, y_bal)
    return clf

# === Evaluation ===
def evaluate(true, pred):
    acc = accuracy_score(true, pred)
    report = classification_report(true, pred, output_dict=True, zero_division=0)['weighted avg']
    return acc, report['f1-score']

# === Single 80/20 Train-Test Pipeline ===
def run_pipeline(taxonomy_path, category_path):
    df = load_data(taxonomy_path, category_path)
    model = SentenceTransformer('all-mpnet-base-v2')

    # Split 80/20
    df_train, df_test = train_test_split(df, test_size=0.20, stratify=df['full_category'], random_state=42)

    # Train
    X_train = embed_text(df_train, model)
    y_train = df_train['full_category']
    clf = train_classifier(X_train, y_train)

    # Test
    X_test = embed_text(df_test, model)
    y_test = df_test['full_category']
    preds = clf.predict(X_test)

    df_test = df_test.copy()
    df_test['Predicted_full_category'] = preds
    df_test[['Pred_CATEGORY1', 'Pred_CATEGORY2', 'Pred_CATEGORY3']] = df_test['Predicted_full_category'].str.split(' > ', expand=True)

    # Evaluate by category levels
    acc1, f1_1 = evaluate(df_test['FTICategory1'], df_test['Pred_CATEGORY1'])
    acc2, f1_2 = evaluate(df_test['FTICategory2'], df_test['Pred_CATEGORY2'])
    acc3, f1_3 = evaluate(df_test['FTICategory3'], df_test['Pred_CATEGORY3'])

    print("\n=== Final 20% Test Set Results ===")
    print(f"CATEGORY1 Accuracy: {acc1:.4f}, Macro F1: {f1_1:.4f}")
    print(f"CATEGORY2 Accuracy: {acc2:.4f}, Macro F1: {f1_2:.4f}")
    print(f"CATEGORY3 Accuracy: {acc3:.4f}, Macro F1: {f1_3:.4f}")

    return clf, model


# In[ ]:


clf, model = run_pipeline("training_set.xlsx", "CATEGORY.csv")


# In[ ]:


import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate(true, pred):
    acc = accuracy_score(true, pred)
    f1 = classification_report(true, pred, output_dict=True, zero_division=0)['weighted avg']['f1-score']
    return acc, f1

def predict_on_test_file(clf, model, test_file_path, output_path="Test_Predictions.xlsx"):
    # Load file
    df_test = pd.read_excel(test_file_path)

    # Preprocess
    df_test = df_test.copy()
    df_test['product_text'] = df_test[['product_desc_1', 'product_desc_2', 'central_description']] \
        .fillna('').astype(str).agg(' '.join, axis=1)
    df_test['product_text'] = df_test['product_text'].apply(preprocess_text)

    # Embed and predict
    X_test = embed_text(df_test, model)
    preds = clf.predict(X_test)

    # Assign predictions
    df_test['Predicted_full_category'] = preds
    df_test['Pred_CATEGORY1'] = None
    df_test['Pred_CATEGORY2'] = None
    df_test['Pred_CATEGORY3'] = None

    for idx, val in df_test['Predicted_full_category'].items():
        parts = str(val).split(' > ')
        if len(parts) > 0:
            df_test.at[idx, 'Pred_CATEGORY1'] = parts[0]
        if len(parts) > 1:
            df_test.at[idx, 'Pred_CATEGORY2'] = parts[1]
        if len(parts) > 2:
            df_test.at[idx, 'Pred_CATEGORY3'] = parts[2]

    # Evaluation
    print("\n=== Evaluation Metrics ===")
    acc1, f1_1 = evaluate(df_test['FTICategory1'], df_test['Pred_CATEGORY1'])
    acc2, f1_2 = evaluate(df_test['FTICategory2'], df_test['Pred_CATEGORY2'])
    acc3, f1_3 = evaluate(df_test['FTICategory3'], df_test['Pred_CATEGORY3'])

    print(f"CATEGORY1 Accuracy: {acc1:.4f}, Weighted F1: {f1_1:.4f}")
    print(f"CATEGORY2 Accuracy: {acc2:.4f}, Weighted F1: {f1_2:.4f}")
    print(f"CATEGORY3 Accuracy: {acc3:.4f}, Weighted F1: {f1_3:.4f}")

    # Save results to Excel
    df_test.to_excel(output_path, index=False)
    print(f"\n✅ Predictions saved to {output_path}")
    return df_test


# In[ ]:


output_file = predict_on_test_file(clf, model, "Test_Set.xlsx")


# In[ ]:


# prompt: real test final with predictions and give records where acural and preficted are not same

# Display records where actual and predicted categories do not match for any level
mismatched_records = output_file[
    (output_file['FTICategory1'] != output_file['Pred_CATEGORY1']) |
    (output_file['FTICategory2'] != output_file['Pred_CATEGORY2']) |
    (output_file['FTICategory3'] != output_file['Pred_CATEGORY3'])
].copy()

# Select relevant columns for display
display_cols = [
    'product_desc_1', 'product_desc_2', 'central_description',
    'FTICategory1', 'FTICategory2', 'FTICategory3',
    'Pred_CATEGORY1', 'Pred_CATEGORY2', 'Pred_CATEGORY3'
]

print("\n=== Records where Actual and Predicted Categories Do Not Match ===")
if not mismatched_records.empty:
    print(mismatched_records[display_cols])
else:
    print("No mismatched records found.")


# In[ ]:


# prompt: write mismatched records to file and save the file

# Define the path for the mismatched records file
mismatched_output_path = "Mismatched_Records.xlsx"

# Save the mismatched records to an Excel file
if not mismatched_records.empty:
    mismatched_records.to_excel(mismatched_output_path, index=False)
    print(f"\n✅ Mismatched records saved to {mismatched_output_path}")
else:
    print("\nNo mismatched records to save.")

