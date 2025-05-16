# Sentiment Analysis with Classical & Deep Learning NLP Models

Welcome to the **Sentiment Analysis NLP Project**!  
This repository demonstrates a comprehensive approach to sentiment analysis on social media text using a progression of NLP techniques-from classic Bag-of-Words and TF-IDF, through distributed word embeddings (Word2Vec), to deep learning models (LSTM) and modern transformer architectures.

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Techniques Used](#techniques-used)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results & Insights](#results--insights)
- [References](#references)

---

## Project Overview

This project explores sentiment analysis using three main types of language models:

1. **Basic Language Models**: Bag-of-Words (BoW), TF-IDF
2. **Distributed Language Models**: Word2Vec embeddings
3. **Context-based Language Models**: LSTM and Transformer (BERT-like) models

The workflow includes:
- Deep-dive preprocessing and cleaning of tweets
- Feature engineering using BoW, TF-IDF, and Word2Vec
- Model building with traditional ML, LSTM, and Transformers
- Evaluation and comparison of approaches

---

## Techniques Used

### 1. **Preprocessing**
- Removal of URLs, special characters, and punctuation
- Lowercasing, tokenization, stopword removal
- Lemmatization

### 2. **Feature Engineering**
- **BoW & TF-IDF**: Classical document-term matrix representations
- **Word2Vec**: Distributed word embeddings (using Gensim)
- **Word Embedding Layer**: For deep learning models

### 3. **Modeling Approaches**
- **Traditional ML**: Logistic Regression, SVM, Random Forest (with BoW/TF-IDF)
- **Deep Learning**: LSTM with embedding layers
- **Transformers**: (e.g., BERT) for context-aware sentiment classification

---

## Project Structure

```
.
├── Part-2-Sentiment-Analysis-bow_tf_idf.ipynb      # BoW & TF-IDF with ML models
├── Part_3_Word2Vec_Embeddings.ipynb                # Word2Vec embeddings & ML/DNN
├── Part_4_LSTM_and_Transformations.ipynb           # LSTM & Transformer models
├── data/
│   └── cleaned_tweets_v1.pkl                       # Preprocessed dataset
└── README.md                                       # Project documentation
```

---

## How to Run

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/sentiment-nlp-project.git
    cd sentiment-nlp-project
    ```

2. **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

3. **Open Jupyter Notebooks**
    - Run each notebook in order:
        1. `Part-2-Sentiment-Analysis-bow_tf_idf.ipynb`
        2. `Part_3_Word2Vec_Embeddings.ipynb`
        3. `Part_4_LSTM_and_Transformations.ipynb`

4. **(Optional) Try the Transformer model**
    - Requires `transformers` and a GPU for best performance.

---

## Results & Insights

| Model Type            | Feature        | Accuracy* | Context Handling |
|-----------------------|---------------|-----------|-----------------|
| Logistic Regression   | TF-IDF        | ~82%      | ❌              |
| DNN                   | Word2Vec      | ~85%      | ⚠️ Limited      |
| LSTM                  | Embedding     | ~88%      | ✅ Sequential   |
| Transformer (BERT)    | Embedding     | ~92%+     | ✅ Full         |

*Results are indicative and may vary depending on tuning and dataset splits.

**Key Findings:**
- Deep preprocessing and feature engineering are critical for model performance.
- Word2Vec and LSTM models outperform classical ML on nuanced sentiment.
- Transformers achieve the best results, especially on complex, context-dependent text.

---

## References

- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [Scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Keras LSTM Layer](https://keras.io/api/layers/recurrent_layers/lstm/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)

---

## 🚀 Future Work

- Hyperparameter tuning and model ensembling
- Deployment as a REST API (FastAPI/Flask)
- Interactive web demo (e.g., Gradio)

---

**Contributions welcome!**  
Feel free to open issues or submit pull requests for improvements.

---

> *Built with ❤️ by [Saikat Pal](https://github.com/saikat912)*

---

**Note:**  
For any questions or suggestions, please open an issue or contact me directly.

---

*Replace placeholder URLs and names with your actual GitHub username and links before publishing!*

