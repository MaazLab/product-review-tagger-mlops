import torch
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# ----- Configuration -----
MODEL_NAME = "nreimers/MiniLM-L6-H384-uncased"
MODEL_PATH = "models/minilm_multilabel.pth"
BINARIZER_PATH = "models/label_binarizer.pkl"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# ----- Load Label Binarizer -----
with open(BINARIZER_PATH, "rb") as f:
    mlb = pickle.load(f)

# ----- Load Tokenizer -----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ----- Load Model -----
model = MiniLMMultiLabelClassifier(MODEL_NAME, len(mlb.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----- Test Input -----
sample_text = "The packaging was damaged and the battery does not last long."

# ----- Tokenization -----
inputs = tokenizer.encode_plus(
    sample_text,
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt"
)

input_ids = inputs["input_ids"].to(DEVICE)
attention_mask = inputs["attention_mask"].to(DEVICE)

# ----- Inference -----
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = outputs.cpu().numpy()[0]
    preds = (probs > 0.5).astype(int)

# ----- Decode Labels -----
predicted_tags = mlb.inverse_transform(np.array([preds]))[0]
# ----- Output -----
print(f"\nInput: {sample_text}")
print("Predicted Tags:", predicted_tags)
