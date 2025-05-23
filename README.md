# 🧠 Intelligent Log Classification System

## 📌 Overview

This project implements a **hybrid log classification system** that combines **rule-based**, **machine learning**, and **large language model (LLM)** techniques to accurately classify system and application log messages into predefined categories.

---

## 🎯 Objectives

- Automatically classify log messages from various sources (e.g., CRM systems, APIs).
- Improve classification accuracy using **Regex**, **BERT-based ML**, and **LLMs** as fallback.
- Support easy integration and extensibility for production-level pipelines.

---

## 🧩 Components

### 1. `training.py`
- Reads a CSV dataset of synthetic logs.
- Uses `all-MiniLM-L6-v2` from **SentenceTransformers** to embed log messages.
- Applies **DBSCAN clustering** to group similar messages.
- Classifies known patterns using **regular expressions**.
- Trains a **Logistic Regression** model on unknown/unlabeled logs.
- Saves the trained model as `log_classifier.joblib`.

---

### 2. `processor_LLM.py`
- Utilizes a Large Language Model (e.g., `deepseek-r1-distill-llama-70b`) via **Groq API**.
- Handles complex or unstructured logs through prompt-based classification.
- Categorizes messages into labels like `Workflow Error`, `Deprecation Warning`, etc.

---

### 3. `processor_bert.py`
- Loads the trained **sentence embedding model** and **classifier**.
- Implements `classify_with_bert()` to predict log categories.
- Returns `"Unclassified"` for low-confidence predictions (probability < 0.5).

---

### 4. `processor_regex.py`
- Implements `classify_with_regex()` using a rule-based approach.
- Designed to detect known log formats.
- Example categories: `User Action`, `System Notification`, etc.

---

### 5. `clarify.py`
- The main controller script.
- Routes logs dynamically:
  - Uses **LLM** for `LegacyCRM` source logs.
  - Applies **Regex first**, then ML fallback for other sources.
- Includes `classify_csv()` to process logs in a CSV and save the classified results.

---

## 📂 Example Usage
```bash
  python clarify.py
```
Processes logs from Input.csv.
Outputs classification results to output.csv.

---

## ✅ Features

- 🔁 **Hybrid Classification Pipeline**  
  Combines **Regex**, **Machine Learning (ML)**, and **Large Language Models (LLMs)** to ensure robust log classification with fallback mechanisms.

- 🤖 **ML-Driven Predictions**  
  Uses `all-MiniLM-L6-v2` sentence embeddings with a **Logistic Regression** classifier for high-confidence predictions.

- 🧠 **LLM-Powered Understanding**  
  Utilizes powerful LLMs (e.g., `deepseek-r1-distill-llama-70b`) via Groq API to interpret and classify complex or legacy log messages.

- 🧱 **Modular Design**  
  Each component (Regex, BERT, LLM) is implemented as a separate module, making the system easy to maintain and extend.

- 🧪 **Clustering for Insight**  
  DBSCAN is used for unsupervised grouping to uncover hidden log patterns.

- 💾 **CSV-Based Input and Output**  
  Accepts structured log data in CSV format and writes back predictions to a new file.

- 🧭 **Source-Specific Routing**  
  Custom logic for routing log messages from different sources (e.g., `LegacyCRM`, `ModernAPI`) through appropriate classifiers.

---

## 📦 Requirements

Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
## 🚀 Usage

Train the model (optional if already trained):
```bash
python training.py
```
Run the classifier on a test file:
```bash
python clarify.py
```
This reads logs from Treaning/test.csv and writes results to output.csv.

## Security Tips

To avoid exposing sensitive credentials:

Add .env to your .gitignore.
Never commit .env or API keys to your public repository.
To remove .env from GitHub history:

```bash
# Install BFG Repo-Cleaner first:
brew install bfg  # macOS
# Then run:
bfg --delete-files .env
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push origin --force
