import torch
import numpy as np
import pickle
from transformers import BertTokenizer
from train import BERTMultiLabelClassifier  # Import from your training script

# Load resources
MODEL_PATH = 'models/bert_multilabel.pth'
TOKENIZER_PATH = 'models/bert_tokenizer'
BINARIZER_PATH = 'models/label_binarizer.pkl'
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 256
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')  # For demonstration purposes, using CPU

# Load tokenizer and label binarizer
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
with open(BINARIZER_PATH, 'rb') as f:
    mlb = pickle.load(f)

# Load model
model = BERTMultiLabelClassifier(MODEL_NAME, len(mlb.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Sample review for testing
sample_text = "The packaging was damaged and the battery does not last long."

# Tokenize
inputs = tokenizer.encode_plus(
    sample_text,
    None,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_token_type_ids=False,
    return_attention_mask=True,
    return_tensors='pt'
)

input_ids = inputs['input_ids'].to(DEVICE)
attention_mask = inputs['attention_mask'].to(DEVICE)

# Predict
with torch.no_grad():
    outputs = model(input_ids, attention_mask)
    predicted_probs = outputs.cpu().numpy()[0]
    predicted_labels = (predicted_probs > 0.5).astype(int)

# Decode
tags = mlb.inverse_transform([predicted_labels])[0]
print(f"\nInput: {sample_text}")
print(f"Predicted Tags: {tags}")
