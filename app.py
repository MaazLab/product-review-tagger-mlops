import torch
import torch.nn as nn
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertModel
import uvicorn

# ----- Configuration -----
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 256
MODEL_PATH = 'models/bert_multilabel.pth'
TOKENIZER_PATH = 'models/bert_tokenizer'
BINARIZER_PATH = 'models/label_binarizer.pkl'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ----- Load Artifacts -----
with open(BINARIZER_PATH, 'rb') as f:
    mlb = pickle.load(f)
num_labels = len(mlb.classes_)

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

model = BERTMultiLabelClassifier(MODEL_NAME, num_labels)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----- FastAPI Setup -----
app = FastAPI(title="Product Review Tagger")

class ReviewRequest(BaseModel):
    review_text: str

@app.get("/")
def home():
    return {"message": "Product Review Tagger API is running."}

@app.post("/predict")
def predict(request: ReviewRequest):
    text = request.review_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty review_text provided.")

    # Tokenization
    inputs = tokenizer.encode_plus(
        text,
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

    # Inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = outputs.cpu().numpy()[0]
        preds = (probs > 0.5).astype(int)

    tags = mlb.inverse_transform(np.array([preds]))[0]

    return {
        "input": text,
        "predicted_tags": list(tags)
    }

# Optional: for running directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
