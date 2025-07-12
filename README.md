# 🧪 Machine Learning Model Playground

An interactive web app built with **Streamlit** that lets you explore, train, and evaluate machine learning models on your own datasets — without writing any code.
Visit the link to use the app:-
https://mlmodelplayground.streamlit.app/
---

## 🚀 Features

- 📁 Upload any `.csv` dataset
- 🧠 Select target and feature columns
- 🛠 Choose from multiple ML models:
  - Logistic Regression
  - Random Forest (Classifier & Regressor)
  - SVM
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
  - Linear Regression
- 🧪 Tune hyperparameters interactively
- 📊 View:
  - Accuracy / MSE / R²
  - Classification report
  - Confusion matrix (with heatmap)
  - ROC curves for both binary and multiclass classification
  - Actual vs Predicted scatter plot for regression

---

## ⚙️ Installation

### 1. Clone the Repository
bash
git clone https://github.com/your-username/ml-model-playground.git
cd ml-model-playground
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Or manually:

bash
Copy
Edit
pip install streamlit pandas scikit-learn matplotlib seaborn
3. Run the App
bash
Copy
Edit
streamlit run app.py
🧠 Example Use Cases
Quickly test ML models on your data

Teach machine learning concepts

Compare models and metrics

Visualize performance interactively

📂 Sample Datasets
Use sample_dataset.csv or large_sample_dataset.csv included in this repo or upload your own.

📌 Notes
Handles both classification and regression

ROC curves are automatically plotted for binary and multiclass tasks

Regression metrics: MSE, R² + Actual vs Predicted scatter plot

📄 License
MIT License. Feel free to use, modify, and share!

🙌 Acknowledgements
Built using:

Streamlit

Scikit-learn

Matplotlib

Seaborn

✍️ Author
Made by Kinshuk Narang
