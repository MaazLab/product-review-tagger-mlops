import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ----- Configuration -----
MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

MODEL_PATH = "models/minilm_multilabel.pth"
TOKENIZER_PATH = MODEL_NAME
BINARIZER_PATH = "models/label_binarizer.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Dataset Preparation -----
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

# ----- Model Definition -----
class MiniLMMultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(MiniLMMultiLabelClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = output.pooler_output if hasattr(output, "pooler_output") else output.last_hidden_state[:, 0]
        output = self.dropout(pooled_output)
        return torch.sigmoid(self.classifier(output))

# ----- Load and Preprocess Data -----
df = pd.read_csv("data/processed_reviews.csv")  # Replace with your cleaned file path
texts = df["review_text"].tolist()
labels = df["tags"].apply(lambda x: x.split(",")).tolist()

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

with open(BINARIZER_PATH, "wb") as f:
    pickle.dump(mlb, f)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

train_dataset = ReviewDataset(X_train, y_train, tokenizer)
val_dataset = ReviewDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ----- Initialize Model -----
model = MiniLMMultiLabelClassifier(MODEL_NAME, len(mlb.classes_))
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

# ----- Training Loop -----
print("Training started...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} finished with loss: {total_loss / len(train_loader):.4f}")

# ----- Save Model -----
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved to:", MODEL_PATH)
