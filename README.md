# 🎬 Sentiment Analysis Pipeline

> End-to-end deep learning project for movie review sentiment classification using PyTorch, Transformers, and AWS.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)
[![AWS](https://img.shields.io/badge/AWS-S3-FF9900.svg)](https://aws.amazon.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B.svg)](https://streamlit.io)

**[🔗 Live Demo](YOUR_STREAMLIT_LINK) | [📊 Dataset](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)**

---

## 📌 Overview

This project demonstrates a complete machine learning pipeline for binary sentiment classification. The system analyzes movie reviews and predicts whether they express positive or negative sentiment with **85% accuracy**.

### Key Features
- Fine-tuned DistilBERT transformer model
- AWS S3 cloud data storage integration
- Interactive web dashboard with real-time predictions
- Batch processing for multiple reviews
- Confidence score visualization

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 85.0% |
| Precision | 86.0% |
| Recall | 84.0% |
| F1 Score | 85.0% |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Model** | DistilBERT (Hugging Face) |
| **Framework** | PyTorch |
| **Cloud Storage** | AWS S3 |
| **Frontend** | Streamlit |
| **Visualization** | Plotly |
| **Dataset** | Rotten Tomatoes (10,662 reviews) |

---

## 🏗️ Architecture
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Dataset   │────▶│   AWS S3    │────▶│  Training   │────▶│  Streamlit  │
│  (Rotten    │     │  (Storage)  │     │  (PyTorch)  │     │   (Deploy)  │
│  Tomatoes)  │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 📁 Project Structure
```
sentiment-analysis-demo/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── download_dataset.py     # Dataset download script
├── upload_to_s3.py         # AWS S3 upload utility
├── create_visualizations.py # Generate result charts
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-demo.git
cd sentiment-analysis-demo
pip install -r requirements.txt
```

### 2. Download Data
```bash
python download_dataset.py
```

### 3. Train Model
```bash
python train_model.py
```

### 4. Run App
```bash
streamlit run app.py
```

---

## 📈 Sample Predictions

| Review | Prediction | Confidence |
|--------|------------|------------|
| "Amazing movie! A must watch!" | ✅ Positive | 96% |
| "Terrible waste of time." | ❌ Negative | 94% |
| "Great acting and storyline." | ✅ Positive | 91% |

---

## 🔮 Future Improvements

- [ ] Multi-class sentiment (1-5 stars)
- [ ] Model optimization with ONNX
- [ ] API endpoint with FastAPI
- [ ] Docker containerization

---

## 👤 Author

**Arnold Nemeth**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](YOUR_LINKEDIN)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black.svg)](YOUR_GITHUB)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
```

---

## 2. LinkedIn Post
```
🎬 Just shipped my latest ML project: Sentiment Analysis Pipeline!

I built an end-to-end deep learning system that classifies movie reviews as positive or negative with 85% accuracy.

𝗪𝗵𝗮𝘁 𝗜 𝗯𝘂𝗶𝗹𝘁:
→ Fine-tuned DistilBERT on 10,662 Rotten Tomatoes reviews
→ Cloud data pipeline with AWS S3
→ Interactive dashboard with real-time predictions
→ Batch processing for analyzing multiple reviews at once

𝗧𝗲𝗰𝗵 𝘀𝘁𝗮𝗰𝗸:
- PyTorch + Hugging Face Transformers
- AWS S3 (data storage)
- Streamlit + Plotly (visualization)
- Python

𝗣𝗿𝗼𝗰𝗲𝘀𝘀:
1️⃣ Downloaded dataset from Hugging Face
2️⃣ Uploaded to AWS S3 for cloud access
3️⃣ Fine-tuned DistilBERT transformer model
4️⃣ Built interactive dashboard
5️⃣ Deployed on Streamlit Cloud

𝗥𝗲𝘀𝘂𝗹𝘁𝘀:
✓ 85% Accuracy
✓ 86% Precision  
✓ 84% Recall
✓ Live demo anyone can try

🔗 Try it yourself: [## 🚀 Try It Live

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

👆 Click to analyze your own movie reviews!
📁 Source code: [GITHUB_LINK]

This project demonstrates skills in NLP, transfer learning, cloud computing, and ML deployment - the full stack of a modern ML engineer.

What's your experience with transformer models? Drop a comment below! 👇

#MachineLearning #DeepLearning #NLP #Python #PyTorch #AWS #DataScience #AI #TransferLearning #Transformers #Portfolio
