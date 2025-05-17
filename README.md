# product-review-tagger-mlops

A Deep Learning project for multi-label classification of product reviews, fully automated with an MLOps workflow. The trained model is packaged into a Docker container and pushed to Docker Hub for easy deployment and inference.

---

## 🚀 Project Overview

This project builds an NLP model that automatically tags product reviews with relevant labels such as:

- Price
- Quality
- Delivery
- Customer Service
- Packaging
- Others (customizable)

The end-to-end pipeline includes:

- Data preprocessing
- Multi-label classification model training (CNN/BERT)
- API deployment using FastAPI
- Dockerization
- CI/CD automation for Docker Hub publishing

---

## 🧠 Model Architecture

- Text preprocessing with tokenization and padding
- Model options:
  - Baseline: CNN or BiLSTM
  - Advanced: BERT-based fine-tuned model
- Output layer: Sigmoid-activated dense layer for multi-label output

---

## 📦 Deployment (MLOps)

The deployment pipeline includes:

1. **Model training** on provided dataset
2. **Docker build** to containerize the API
3. **Push to Docker Hub** using GitHub Actions
4. **Pull and run** the image anywhere using Docker CLI

---

## 🛠 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/your-username/product-review-tagger-mlops.git
cd product-review-tagger-mlops

# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Train the model
python train.py

# Run the FastAPI app
uvicorn app.main:app --reload
```

---

## 🐳 Docker Usage

```bash
# Build Docker image
docker build -t your-username/product-review-tagger .

# Run container
docker run -p 8000:8000 your-username/product-review-tagger

# Access API at
http://localhost:8000/docs

```

---

## 🤖 CI/CD Pipeline

- Configured with **GitHub Actions**
- Automatically builds and pushes Docker image to Docker Hub on every push to `main` branch

---

## 📊 Dataset

- Source: [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- Format: Text reviews with associated tags
-  Preprocessing includes tokenization, stopword removal, and label binarization

---

## 📈 Evaluation Metrics

- Hamming Loss
- F1 Score (Micro/Macro)
- Precision@K
- ROC-AUC

---

## 📌 To-Do

- [ ] Add model evaluation script
- [ ] Integrate BERT model option
- [ ] Enable dynamic label detection
- [ ] Add monitoring and logging (future MLOps enhancement)

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Maintainer

[Maaz Rafiq]  
Email: [Maaz.rafique.75@gmail.com]  
GitHub: [@MaazLab](https://github.com/MaazLab)
