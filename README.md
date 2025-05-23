 # Intelligent Log Classification System

# 📌 Overview
This project implements a hybrid log classification system that combines rule-based, machine learning, and large language model (LLM) techniques to accurately classify system and application log messages into predefined categories.

# 🎯 Objectives
Automatically classify log messages from various sources such as CRM systems and APIs.
Improve classification accuracy using regex, BERT-based ML, and LLMs for fallback.
Support easy integration and model extensibility for production pipelines.
# 🧠 Components
1. training.py

Reads a CSV dataset of synthetic logs.
Uses the all-MiniLM-L6-v2 SentenceTransformer to embed log messages.
Applies DBSCAN clustering to group similar log messages.
Uses regular expressions to classify known patterns.
Trains a Logistic Regression model on unlabeled and non-legacy logs.
Saves the trained model (log_classifier.joblib) for future inference.
2. process.py

Uses a Large Language Model (e.g., deepseek-r1-distill-llama-70b) via the Groq API.
Classifies complex/unstructured logs into categories like Workflow Error or Deprecation Warning using prompts and pattern matching.
3. process_bart.py

Loads the trained sentence embedding model and the Logistic Regression classifier.
Provides a function classify_with_bert() to predict log categories.
Returns "Unclassified" for low-confidence predictions (< 0.5 probability).
4. process_ragx.py

Implements classify_with_regex() using regex rules for known log message formats.
Categories include: User Action, System Notification, etc.
5. clarify.py

Main controller script.
Dynamically routes log messages through:
LLM for LegacyCRM source logs.
Regex first, then ML fallback for other sources.
Contains classify_csv() to process a CSV and save classified output.
📂 Example Usage
python clarify.py
Processes test.csv logs and outputs output.csv with classified labels.

✅ Features
Hybrid classification logic combining Regex, BERT + Logistic Regression, and LLM fallback.
Modular components with fallback hierarchy.
Handles edge cases and noisy log entries robustly.
Built-in clustering to explore log groups.
Easily extendable for other models and data sources.
📦 Requirements
Python 3.8+
sentence-transformers, scikit-learn, pandas, joblib, dotenv, groq
📈 Example Labels
User Action
System Notification
Workflow Error
Deprecation Warning
Unclassified
📑 Folder Structure (Suggested)
├── training.py
├── process.py
├── process_bart.py
├── process_ragx.py
├── clarify.py
├── dataset/
│   └── synthetic_logs.csv
├── Treaning/
│   └── test.csv
├── output.csv
└── requirements.txt
