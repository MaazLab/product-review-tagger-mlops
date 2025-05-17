import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import json
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
    multilabel_confusion_matrix
)
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# Configuration
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 256
BATCH_SIZE = 8
MODEL_PATH = 'models/bert_multilabel.pth'
TOKENIZER_PATH = 'models/bert_tokenizer'
BINARIZER_PATH = 'models/label_binarizer.pkl'
TEST_DATA_PATH = 'data/processed_reviews.csv'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = 'evaluation_report.json'

# ----- Model Definition -----
class BERTMultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BERTMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.dropout(pooled_output)
        return torch.sigmoid(self.classifier(output))

# Load binarizer and tokenizer
with open(BINARIZER_PATH, 'rb') as f:
    mlb = pickle.load(f)
classes = mlb.classes_
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
num_labels = len(classes)

# Load model
model = BERTMultiLabelClassifier(MODEL_NAME, num_labels)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load and prepare test data
df = pd.read_csv(TEST_DATA_PATH)
df['tags'] = df['tags'].apply(lambda x: x.split(','))
df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

X = df['review_text']
y = mlb.transform(df['tags'])

def tokenize_batch(texts):
    return tokenizer(
        list(texts),
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )

# Batched prediction
all_preds, all_labels = [], []
for i in tqdm(range(0, len(X), BATCH_SIZE)):
    batch_texts = X[i:i + BATCH_SIZE]
    batch_labels = y[i:i + BATCH_SIZE]

    inputs = tokenize_batch(batch_texts)
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask).cpu().numpy()

    all_preds.extend(outputs)
    all_labels.extend(batch_labels)

# Convert to NumPy arrays
y_true = np.array(all_labels)
y_scores = np.array(all_preds)
y_pred = (y_scores > 0.5).astype(int)

# ----- Metrics -----
report = {}

# Hamming Loss
report['hamming_loss'] = hamming_loss(y_true, y_pred)

# F1 Scores
report['f1_micro'] = f1_score(y_true, y_pred, average='micro')
report['f1_macro'] = f1_score(y_true, y_pred, average='macro')

# Precision@K
def precision_at_k(y_true, y_scores, k=3):
    top_k_preds = np.argsort(-y_scores, axis=1)[:, :k]
    hits = 0
    for i in range(len(y_true)):
        true_labels = set(np.where(y_true[i])[0])
        pred_top_k = set(top_k_preds[i])
        hits += len(true_labels & pred_top_k) / k
    return hits / len(y_true)

report['precision_at_3'] = precision_at_k(y_true, y_scores, k=3)

# ROC-AUC
try:
    report['roc_auc_macro'] = roc_auc_score(y_true, y_scores, average='macro')
except:
    report['roc_auc_macro'] = "Unavailable (missing labels in test set)"

# Per-label precision, recall, f1
per_label_report = classification_report(
    y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
)
report['per_label_metrics'] = {
    label: {
        'precision': round(per_label_report[label]['precision'], 4),
        'recall': round(per_label_report[label]['recall'], 4),
        'f1-score': round(per_label_report[label]['f1-score'], 4)
    }
    for label in classes
}

# Optional: Save multilabel confusion matrix
conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
np.save("confusion_matrices.npy", conf_matrix)

# Save report
with open(SAVE_PATH, 'w') as f:
    json.dump(report, f, indent=4)

# Print summary
print("\n=== Evaluation Summary ===")
for key, value in report.items():
    if key == "per_label_metrics":
        print("\nPer-label scores:")
        for label, metrics in value.items():
            print(f"  {label}: {metrics}")
    else:
        print(f"{key}: {value}")
print("\n=== Evaluation report saved to", SAVE_PATH, "===")