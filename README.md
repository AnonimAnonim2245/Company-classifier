# Company Classification System

## Project Overview
This project implements a **large-scale AI-powered classification system** for categorizing companies based on their descriptions. It leverages **FAISS (Facebook AI Similarity Search), Context Similarity (BERT, TF-IDF, Cosine Similarity), and Machine Learning techniques** to efficiently classify 9,500+ companies across 220 categories.

## Key Features
- **Efficient FAISS-based vector search** for rapid retrieval.
- **Hybrid classification system** integrating **TF-IDF, Word Embeddings (Word2Vec, BERT), and FAISS indexing**.
- **Reranking with BERT embeddings** for improved accuracy.
- **Optimized search latency** reducing query time from **seconds to milliseconds**.
- **Custom pipeline for multi-step text similarity and classification.**


## Technologies Used
- **Python** (Numpy, Pandas, Scikit-learn)
- **FAISS** (Efficient Nearest Neighbor Search)
- **TF-IDF** (Text Frequency - Inverse Document Frequency)
- **Word2Vec / BERT** (Contextual Word Embeddings)
- **Cosine Similarity & Context Similarity**
- **Machine Learning (Logistic Regression, Random Forest, Transformers)**

---

## Steps in the Classification Pipeline
### **1️. Data Preprocessing**
- Loaded **company descriptions** and **taxonomy labels**.
- Tokenized, lemmatized, and removed stopwords.
- Converted text into **TF-IDF weighted features**.

### **2. FAISS Indexing & Nearest Neighbor Search**
- Created **word embeddings** using **BERT & Word2Vec**.
- Built **FAISS index** to store vectorized company descriptions.
- Implemented **fast nearest neighbor search**.

### **3️. Context Similarity Calculation**
- Computed **Cosine Similarity** between companies and labels.
- Used **BERT sentence embeddings** to refine classification.

### **4️. Hybrid Classification & Reranking**
- FAISS retrieved **Top-N similar labels**.
- Applied **BERT reranking** to re-evaluate the closest matches.
- Combined **TF-IDF + FAISS + Context Similarity** for final classification.

### **5️. Model Evaluation & Optimization**
- Measured accuracy and precision using **classification metrics**.
- Optimized FAISS parameters for better search speed.
- Improved accuracy using **ensemble methods** combining multiple ML techniques.

