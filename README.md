# ❤️ Heart Disease Risk Predictor (ML + Streamlit)

## 🚀 Overview

This project builds an end-to-end machine learning pipeline to predict heart disease risk using clinical data.

Instead of just training a model, this project delivers a **working product**:

* ML model (Random Forest)
* Saved pipeline (.pkl)
* Interactive Streamlit UI

---

## 🧠 Model

* Algorithm: Random Forest Classifier
* Trained on: UCI Heart Disease dataset
* Output:

  * Probability of heart disease
  * Risk classification:

    * 🟢 Low Risk
    * 🟠 Medium Risk
    * 🔴 High Risk

⚠️ Custom threshold used:

* **0.20 (instead of default 0.5)**
  → improves early risk detection (higher recall)

---

## ⚙️ Tech Stack

* Python (pandas, scikit-learn)
* Streamlit (UI)
* Joblib (.pkl model)

---

## 📊 Demo (Run locally)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
heart-disease-ml/
│
├── streamlit_app.py          # Streamlit UI
├── heart_rf_pipeline.pkl     # Trained model (Random Forest pipeline)
├── requirements.txt
├── README.md
```

---

## 🧪 Example Output

* Probability: 0.72
* Risk Level: 🔴 High Risk

---

## 💡 Key Highlights

* End-to-end ML workflow (data → model → product)
* Custom threshold tuning (business-oriented decision)
* Simple UI → usable by non-technical users
* Ready for deployment (Streamlit / API)

---

## 📌 Why this project matters

This project demonstrates:

* Ability to move beyond notebooks
* Understanding of real-world ML deployment
* Clear communication of model outputs to users

---

## 👤 Author

**Binh Vo**
📍 Auckland, New Zealand

GitHub: https://github.com/binhvo9
Portfolio: https://binhportfolio-production.up.railway.app
LinkedIn: https://linkedin.com/in/binh-vo-nz
