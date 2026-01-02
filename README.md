# 💰 Income Prediction Web App (Streamlit + Machine Learning)

A **Machine Learning–powered Streamlit web application** that predicts whether a person earns **more than $50K or ≤ $50K per year** based on demographic, education, and work-related features.

This project demonstrates an **end-to-end ML workflow**, including model loading, preprocessing, user interaction, and real-time prediction.

---

## 🚀 Features

* 📊 Interactive **Streamlit web interface**
* 🤖 Pre-trained **Random Forest Classifier**
* 🧠 Handles both **numerical and categorical features**
* 🔁 Automatic **one-hot encoding & feature alignment**
* 📈 Displays **prediction + probability scores**
* 🧩 Clean, modular, and extensible codebase

---

## 🛠️ Tech Stack

| Component     | Technology                  |
| ------------- | --------------------------- |
| Language      | Python                      |
| Web Framework | Streamlit                   |
| ML Model      | Random Forest Classifier    |
| Model Storage | Joblib                      |
| Data Handling | Pandas                      |
| Dataset       | Adult Census Income Dataset |

---

## 📂 Project Structure

```
├── app.py                  # Streamlit application
├── income_model.pkl        # Trained Random Forest model
├── main.ipynb              # Model training & experimentation
├── README.md               # Project documentation
```

---

## 🧾 Input Features

The app collects the following details from the user:

### 🔢 Numerical Features

* Age
* Final Weight (fnlwgt)
* Education Number
* Capital Gain
* Capital Loss
* Hours per Week

### 🔤 Categorical Features

* Workclass
* Education
* Marital Status
* Occupation
* Relationship
* Race
* Gender
* Native Country

All categorical features are **one-hot encoded** and automatically aligned with the model’s training schema.

---

## 🔮 Prediction Output

* **Income Category**

  * `>50K`
  * `<=50K`
* **Prediction Probability**

  * Confidence score for both classes

![image alt](https://github.com/Mahek1305/INCOME-PREDICTION/blob/main/Screenshot%202026-01-02%20135051.png)
![image alt](https://github.com/Mahek1305/INCOME-PREDICTION/blob/main/Screenshot%202026-01-02%20135102.png)

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/income-prediction-streamlit.git
cd income-prediction-streamlit
```

### 2️⃣ Install Dependencies

```bash
pip install streamlit pandas scikit-learn joblib
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser.

---

## 🧠 Model Details

* **Algorithm**: Random Forest Classifier
* **Training Notebook**: `main.ipynb`
* **Model File**: `income_model.pkl`
* **Preprocessing**:

  * One-hot encoding
  * Feature alignment using `feature_names_in_`

This ensures **no mismatch** between training and inference features.

---

## 📌 Use Cases

* ML & Data Science academic project
* Resume/portfolio project for **product-based companies**
* Streamlit dashboard demonstration
* Binary classification case study

---

## 🔮 Future Enhancements

* ✅ Model performance metrics (Accuracy, ROC, Confusion Matrix)
* 🌐 Deploy on Streamlit Cloud / AWS / Render
* 📊 Add feature importance visualization
* 🧪 Support multiple ML models
* 💾 Save prediction history

---

## 👤 Author

**Mahek**
AI & Data Science Student
Focused on **Machine Learning, Data Science & Product-Based ML Projects**



