# 🚨 Fraudulent Job Posting Classification

> Machine Learning & NLP project for detecting fake job postings using text preprocessing, feature engineering, and classification models.

---

## 📌 Overview

Online recruitment platforms contain thousands of job postings daily, but many of them are fraudulent and designed to scam applicants.

This project builds a **Fraudulent Job Detection System** capable of classifying job advertisements into:

* ✅ Real Job Posting
* ❌ Fraudulent Job Posting

using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

---

## ✨ Features

* 🔍 Fraudulent job detection
* 🧹 NLP text preprocessing
* 📊 Exploratory Data Analysis (EDA)
* 🧠 Machine Learning classification models
* 📈 Model evaluation metrics

---

## 🗂️ Dataset

Dataset: **Fake Job Postings Dataset**

### Main Features

* title
* company_profile
* description
* requirements
* benefits
* industry
* employment_type
* required_experience
* required_education
* fraudulent (Target)

### 🎯 Target Labels

| Label | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate Job Posting |
| 1     | Fraudulent Job Posting |

---

# ⚙️ Tech Stack

## 🐍 Language

* Python

## 📚 Libraries

* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn


---

# 🔄 Project Workflow

## 1️⃣ Data Preprocessing

* Handling missing values
* Removing duplicates
* Lowercasing text
* Removing punctuation
* Stopwords removal
* Tokenization
* Text cleaning using Regex

---

## 2️⃣ Exploratory Data Analysis (EDA)

Performed analysis on:

* Fraudulent vs real jobs distribution
* Missing values
* Most common industries
* Word frequency analysis
* Text length analysis

---

## 3️⃣ Feature Engineering

Used NLP techniques such as:

* TF-IDF Vectorization
* Count Vectorization
* Text feature extraction

Combined columns:

* title
* description
* requirements
* company_profile

---

## 4️⃣ Model Training

### Machine Learning Models

* Logistic Regression
* Naive Bayes
* Random Forest
* SVM

### Deep Learning Models

* LSTM
* Embedding Layers

---

# 📈 Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

### Example Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 96%      |
| Random Forest       | 97%      |
| LSTM                | 98%      |

---



---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/ironomar0101-glitch/job_posting_classifcation.git
```

Move into the project directory:

```bash
cd job_posting_classifcation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run The Project

## Run Flask App

```bash
python app.py
```

Application will run on:

```bash
http://127.0.0.1:5000/
```

---

# 🧪 Example Prediction

### Input

```python
"Earn $5000 weekly from home with no experience needed"
```

### Output

```python
Fraudulent Job Posting
```

---

# 🚀 Future Improvements

* Add BERT / Transformers
* Deploy using Docker
* Build REST API
* Improve class imbalance handling
* Add Streamlit dashboard

---

# 📚 Key Learnings

* NLP preprocessing
* Text vectorization
* Binary classification
* End-to-end ML pipeline building

---

# 👨‍💻 Author

## Khaled Omar

* 💻 Machine Learning & AI Enthusiast
* 🔗 GitHub: [https://github.com/ironomar0101-glitch](https://github.com/ironomar0101-glitch)
