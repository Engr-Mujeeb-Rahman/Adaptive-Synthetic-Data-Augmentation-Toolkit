# 🚀 Adaptive Synthetic Data Augmentation Toolkit

A machine learning project focused on improving fraud detection performance using **synthetic data augmentation techniques** and an interactive **Streamlit web application**.

---

## 📌 Project Overview

Fraud detection datasets are highly imbalanced, where fraudulent cases are rare compared to normal transactions. This leads to biased models that fail on minority classes.

This project solves that by:
- Applying synthetic data augmentation techniques
- Training a fraud detection model on improved data distribution
- Comparing model performance before and after augmentation
- Deploying a Streamlit-based prediction interface

---

## 📁 Repository Structure

```
Adaptive-Synthetic-Data-Augmentation-Toolkit/
│
├── Notebook/                  # Jupyter notebook(s) for EDA, training, augmentation
│
├── app.py                     # Streamlit web application
│
├── final_model.pkl           # Trained ML model
├── features.pkl              # Feature columns used in training
├── training_history.csv      # Model training logs / performance tracking
│
├── background.jpg            # UI background image
├── logo.webp                 # Project logo
│
├── Problem Statement.pdf     # Problem definition and task details
├── README.md                 # Project documentation
└── .gitignore
```

---

## 🧠 Workflow

### 1. Data Analysis (Notebook/)
- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Understanding class imbalance

### 2. Synthetic Data Augmentation
- Generating additional samples for minority class
- Improving dataset balance for better learning

### 3. Model Training
- Training classification model on processed dataset
- Saving trained model (`final_model.pkl`)
- Saving feature schema (`features.pkl`)

### 4. Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Logged results in `training_history.csv`

### 5. Deployment (Streamlit - app.py)
- Loads trained model
- Accepts user input
- Predicts fraud vs non-fraud
- Displays results in a simple UI

---

## 🛠️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Streamlit
- Joblib

---

## ▶️ How to Run This Project

### 1. Clone Repository
```bash
git clone https://github.com/Engr-Mujeeb-Rahman/Adaptive-Synthetic-Data-Augmentation-Toolkit.git
cd Adaptive-Synthetic-Data-Augmentation-Toolkit
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App
```bash
streamlit run app.py
```

---

## 📊 Results

### Before Augmentation
- Recall (Fraud Class): 72%
- F1-Score: 0.83

### After Augmentation
- Recall (Fraud Class): 76%
- F1-Score: 0.85

### Key Improvements
- Better detection of minority (fraud) class  
- Reduced false negatives

---

## 📂 Notebook Usage

Open the Jupyter Notebook for full pipeline:

```bash
Notebook/
```

Inside it you will find:
- Data preprocessing steps
- Synthetic augmentation logic
- Model training
- Evaluation metrics

Run using:

```bash
jupyter notebook
```

---

## 🎯 Use Cases

- Fraud detection systems
- Banking transaction monitoring
- Imbalanced classification problems
- Risk analysis systems

---

## 🔮 Future Improvements

- Integrate SMOTE / GAN-based augmentation
- Add explainable AI (SHAP/LIME)
- Deploy as REST API (FastAPI/Flask)
- Improve UI design of Streamlit app
- Add real-time data streaming support

---

## 👤 Author

**Muhammad Mujeeb Ur Rahman**  
AI Engineer
