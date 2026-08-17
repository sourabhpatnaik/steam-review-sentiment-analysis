# 🎮 SteamSentiment AI — Steam Review Sentiment Analysis

An end-to-end NLP project that classifies Steam game reviews as **Positive** or **Negative** using Word2Vec embeddings and XGBoost, deployed as an interactive Streamlit app.


---

## 📌 Overview

This project covers the complete NLP workflow — from raw review text to a deployed, production-style app:

1. Text preprocessing & tokenization
2. Word2Vec embeddings (200-dim document vectors)
3. Feature scaling
4. Model benchmarking across 7 classifiers
5. Deployment via Streamlit, with single-review and batch CSV modes

## ✨ Features

- **🔍 Single Review Analysis** — paste a review (or pick from sample reviews) and get an instant sentiment prediction with confidence score
- **📁 Batch Analysis (CSV)** — upload a CSV of reviews, pick the text column, and classify hundreds of reviews at once with a downloadable results file
- **📊 Model Information** — Word2Vec config, preprocessing pipeline, and final performance metrics
- **🏆 Model Comparison** — accuracy/F1 comparison across all 7 evaluated models with charts
- **📌 Project Overview** — dataset details and end-to-end pipeline breakdown

## 🧠 Model & Performance

| Metric | Score |
|---|---|
| Accuracy | 90.14% |
| Macro F1 | 0.82 |
| Feature Representation | Word2Vec (200-dim) |
| Final Model | XGBoost |

Because the dataset is imbalanced (~80% positive / 20% negative), **Macro F1** was prioritized over raw accuracy when selecting the final model.

### Models Evaluated

Logistic Regression, Gaussian Naive Bayes, KNN, Decision Tree, Random Forest, **XGBoost (selected)**, and an Artificial Neural Network — XGBoost was chosen for the best balance of accuracy, Macro F1, and minority-class recall.

## 🛠️ Tech Stack

- **Language:** Python
- **NLP:** NLTK, Gensim (Word2Vec)
- **ML:** Scikit-learn, XGBoost
- **App/UI:** Streamlit
- **Data:** Pandas, NumPy

## 📂 Project Structure

```
.
├── app.py                  # Streamlit app (UI + inference)
├── preprocessing.py        # Text cleaning / tokenization (cleaning_text)
├── Final_model/
│   ├── xgb_model.pkl       # Trained XGBoost classifier
│   ├── word2vec.model      # Trained Word2Vec embeddings
│   └── scaler.pkl          # Feature scaler
├── notebooks/               # Model training & evaluation (if applicable)
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/sourabhpatnaik/PLACEHOLDER-REPO.git
cd PLACEHOLDER-REPO
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 📊 Using Batch Analysis

1. Go to the **📁 Batch Analysis (CSV)** page
2. Upload a CSV containing a column of review text
3. Select which column holds the reviews
4. Choose how many rows to analyze
5. Run the analysis and download the results as a CSV (includes prediction, confidence, and per-class probabilities for every row)

## 📈 Word2Vec Configuration

| Parameter | Value |
|---|---|
| Vector Size | 200 |
| Window | 10 |
| Epochs | 50 |
| Workers | 24 |

Each review is represented as the **average of its known word vectors**, producing a single 200-dimensional document vector fed into the classifier.

## 🔮 Future Improvements

- Add sentence-transformer / BERT-based embeddings for comparison against Word2Vec
- Support multi-language reviews
- Add explainability (e.g. SHAP) to show which words drove a prediction

## 👤 Author

**Sourabh Patnaik**
[LinkedIn](https://linkedin.com/in/sourabhpatnaik/) · [GitHub](https://github.com/sourabhpatnaik)
