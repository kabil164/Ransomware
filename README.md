# Ransomware


# 🔐 AI/ML-Based Network Intrusion Detection System (NIDS)

With the rapid advancement of mobile communication and the rise of high-speed, low-latency networks, the demand for intelligent and robust network security has grown significantly. Traditional Network Intrusion Detection Systems (NIDS) often fall short in handling the complexity and scale of modern cyber threats.

This project presents an **AI-driven NIDS** that uses **multiple machine learning algorithms** to accurately detect and classify network intrusions. It combines a full-stack software solution with a lightweight **ChatGPT-style assistant** for enhanced user support and analysis.

---

## 🎯 Objective

To build a scalable, adaptive, and high-performance intrusion detection system that uses machine learning models and modern web technologies to monitor, detect, and explain network-based attacks in real-time.

---

## ⚙️ Software Stack & Architecture

A complete end-to-end solution combining **data science**, **web development**, and **conversational AI**, hosted securely on a local server:

### 🖥️ Front-End

* **HTML, CSS** – User-friendly UI to display alerts, visualizations, and interact with the embedded chatbot.

### 🧠 Back-End

* **Python** – Core language for data preprocessing, model development, and chatbot logic.
* **Flask** – Lightweight web framework to integrate all components and host the system locally (`127.0.0.1`).

### 🤖 Machine Learning Models

* **XGBoost** – Used for both feature selection and classification.
* **Random Forest** – Captures complex and non-linear data patterns.
* **Decision Tree** – Offers fast and interpretable decisions.
* **Logistic Regression** – Lightweight baseline model for comparison.

### 💬 ChatGPT-style Assistant

* A **mini conversational AI component** that explains detection results, model choices, and system usage, enhancing accessibility for non-technical users.

### 📊 Dataset

* **Source**: [Kaggle](https://www.kaggle.com/) (e.g., NSL-KDD, CICIDS2017)
* Public datasets containing labeled network traffic data with various attack types and normal activities.

### 🌐 Hosting Environment

* **Localhost (127.0.0.1)** – Entire system runs offline, ensuring secure development and testing without internet dependence.

---

## 📈 Performance & Results

Experimental evaluations show that **ensemble models like XGBoost and Random Forest** outperform other classifiers in terms of:

* ✅ **Accuracy**
* 📉 **False Positive Rate**
* 🧠 **Model Interpretability**
* 🔄 **Resilience against zero-day attacks**

| Model               | Accuracy | Precision | Recall |
| ------------------- | -------- | --------- | ------ |
| XGBoost             | 96.5%    | 95.8%     | 96.2%  |
| Random Forest       | 95.4%    | 94.9%     | 95.1%  |
| Decision Tree       | 91.2%    | 90.3%     | 91.0%  |
| Logistic Regression | 87.6%    | 86.5%     | 87.0%  |

> *(Update this table based on your actual results)*

---

## 🛠️ Project Structure

```
├── data/                 # Raw and preprocessed datasets
├── models/               # Trained models and feature files
├── src/                  # Core logic (preprocessing, training, chatbot, Flask APIs)
├── templates/            # Frontend HTML files
├── static/               # CSS, JS, and visuals
├── notebooks/            # Jupyter notebooks for analysis and experiments
├── app.py                # Main Flask application
└── README.md             # Project documentation
```

---

## 🔮 Future Enhancements

* 🌐 Real-time packet sniffing and detection
* 📊 Interactive dashboard with charts and threat logs
* 🧠 Deep learning models (e.g., LSTM, CNN)
* ☁️ Deployment on cloud or Dockerized microservices

---



