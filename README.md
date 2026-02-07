# Heart Disease Prediction Using Machine Learning

A machine learning project focused on predicting the likelihood of heart disease based on medical attributes. This project explores the complete data science workflow — from data analysis and visualization to model training, tuning, and evaluation — using Python and popular ML libraries.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help in timely medical intervention and significantly improve patient outcomes.

In this project, multiple machine learning models are trained and compared to determine the most effective approach for predicting heart disease using structured health data.

The notebook walks through:

* Data loading and exploration
* Exploratory Data Analysis (EDA)
* Feature understanding
* Model training
* Hyperparameter tuning
* Model evaluation
* Feature importance analysis

---

## 📊 Dataset

The dataset used contains several clinical parameters such as:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol levels
* Maximum heart rate
* Exercise-induced angina
* ST depression
* Number of major vessels
* Thalassemia

**Target Variable:**

* `0` → No heart disease
* `1` → Presence of heart disease

---

## 🧠 Machine Learning Models Used

The following models were implemented and compared:

* **Logistic Regression** – Strong baseline model for classification problems
* **K-Nearest Neighbors (KNN)** – Distance-based learning approach
* **Random Forest Classifier** – Ensemble model for improved predictive performance

---

## ⚙️ Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 🔎 Project Workflow

### 1. Data Exploration

* Checked dataset shape and structure
* Identified class distribution
* Visualized important feature relationships

### 2. Exploratory Data Analysis

* Used statistical summaries
* Created plots to understand trends
* Investigated correlations between variables

### 3. Model Training

A helper function was created to:

* Fit multiple models
* Evaluate them on test data
* Compare performance efficiently

### 4. Hyperparameter Tuning

To improve model performance:

* **RandomizedSearchCV** was used for faster tuning
* **GridSearchCV** was applied for deeper optimization

### 5. Model Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* Cross-validation scores

### 6. Feature Importance

Analyzed which medical features contributed most toward predictions, helping improve interpretability.

---

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/heart-disease-ml.git
cd heart-disease-ml
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Launch the Notebook

```bash
jupyter notebook Heart_Disease_Project.ipynb
```

---

## 📁 Project Structure

```
├── Heart_Disease_Project.ipynb   # Main notebook
├── heart-disease.csv            # Dataset
└── README.md                    # Project documentation
```

---

## 🎯 Key Takeaways

* Proper EDA significantly improves model understanding.
* Hyperparameter tuning can noticeably boost performance.
* Logistic Regression remains a powerful and interpretable baseline for medical classification tasks.
* Feature importance helps bridge the gap between ML predictions and real-world medical insights.

---

## 🔮 Future Improvements

* Deploy the model using Flask or FastAPI
* Build a simple web interface for predictions
* Try advanced models like XGBoost or LightGBM
* Perform feature engineering
* Add model explainability tools such as SHAP

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

If you find this project helpful, consider giving it a ⭐.

---

## 📬 Contact

Feel free to connect if you'd like to discuss machine learning, data science, or improvements to this project.

