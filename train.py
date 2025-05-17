import torch
import torch.nn as nn
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel

# ----- Configuration -----
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 256
MODEL_PATH = 'models/bert_multilabel.pth'
TOKENIZER_PATH = 'models/bert_tokenizer'
BINARIZER_PATH = 'models/label_binarizer.pkl'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

# ----- Load Label Binarizer -----
with open(BINARIZER_PATH, 'rb') as f:
    mlb = pickle.load(f)
num_labels = len(mlb.classes_)

# ----- Load Tokenizer -----
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

# ----- Load Trained Model -----
model = BERTMultiLabelClassifier(MODEL_NAME, num_labels)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----- Inference Input -----
sample_text = "The battery life is terrible and the packaging was ripped."

# ----- Tokenization -----
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

# ----- Inference -----
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = outputs.cpu().numpy()[0]
    preds = (probs > 0.5).astype(int)

# ----- Decode Predicted Tags -----
predicted_tags = mlb.inverse_transform(np.array([preds]))[0]

# ----- Output Result -----
print("\nReview Text:")
print(sample_text)
print("\nPredicted Tags:")
print(predicted_tags)
