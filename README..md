# 🏦 Credit Risk AI App (Agentic AI System)

## 📌 Overview

This project is an **AI-powered credit risk assessment system** that predicts whether a borrower is likely to default and generates a **detailed, explainable report**.

It goes beyond traditional ML by combining:

* Machine Learning (Random Forest)
* LLM (Groq - Llama 3)
* RAG (ChromaDB + Knowledge Base)
* Multi-Agent System (LangGraph)

---

## 📁 Project Structure

```bash
credit-risk-app-endsem/
│
├── app.py               # Main Streamlit app (UI + logic)
├── googlecollab.py      # Original Colab code (reference)
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 🚀 Features

* ✅ Single Borrower Risk Assessment
* ✅ AI-generated Risk Report
* ✅ Risk Classification (LOW → CRITICAL)
* ✅ Red Flag Detection
* ✅ Policy-based Reasoning (RAG)
* ✅ Multi-Agent Workflow

---

## 🧠 System Workflow

```text
User Input
   ↓
Agent 1 → ML Prediction + Risk Metrics
   ↓
Agent 2 → Policy Check (RAG)
   ↓
Agent 3 → Report Generation (LLM)
   ↓
Final Decision + Detailed Report
```

---

## 🤖 Tech Stack

* **Frontend:** Streamlit
* **LLM:** Groq (Llama 3.3 70B)
* **Framework:** LangChain + LangGraph
* **Vector DB:** ChromaDB
* **Embeddings:** HuggingFace
* **ML Model:** Random Forest (Scikit-learn)

---

## 📊 Model Details

* Algorithm: Random Forest
* Accuracy: ~93%
* Input Features: 11 borrower attributes
* Output:

  * 0 → No Default
  * 1 → Default

---

## 📚 How It Works

1. User enters borrower details
2. ML model predicts default probability
3. RAG retrieves relevant credit policies
4. LLM analyzes compliance and risk
5. Final report is generated with:

   * Decision (Approve / Reject)
   * Risk Level
   * Red Flags
   * Recommendations

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Open in browser

```
http://localhost:8501
```

---

## ⚠️ Notes

* Make sure your **API key (Groq)** is set in the code
* Ensure all dependencies are installed
* Python 3.10 / 3.11 recommended

---

## 🎯 Conclusion

This project demonstrates a **real-world AI system** combining:

* ML prediction
* RAG-based knowledge retrieval
* LLM reasoning
* Multi-agent orchestration

👉 Result: A **smart, explainable credit risk analysis tool**

---
