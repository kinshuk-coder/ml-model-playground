# Machine Learning Model Playground (Streamlit + Scikit-Learn)

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Title and description
st.title("🧪 Machine Learning Model Playground")
st.markdown("""
Upload your dataset, select features, choose a model, and tune hyperparameters — all in one interactive interface.
""")

# Upload CSV file
st.sidebar.header("1. Upload Your CSV File")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset:")
    st.dataframe(df.head())

    # Feature and target selection
    st.sidebar.header("2. Select Features and Target")
    columns = df.columns.tolist()
    target_column = st.sidebar.selectbox("Select target column", columns)
    feature_columns = st.sidebar.multiselect("Select feature columns", [col for col in columns if col != target_column])

    if feature_columns:
        X = df[feature_columns]
        y = df[target_column]

        # Split data
        st.sidebar.header("3. Train-Test Split Ratio")
        test_size = st.sidebar.slider("Test size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model selection
        st.sidebar.header("4. Choose Model")
        model_name = st.sidebar.selectbox("Model", [
            "Logistic Regression", "Random Forest", "SVM", "Naive Bayes", "KNN",
            "Linear Regression", "Random Forest Regressor"
        ])

        def get_model(name):
            if name == "Logistic Regression":
                c = st.sidebar.slider("C (Inverse of regularization)", 0.01, 10.0, 1.0)
                return LogisticRegression(C=c, max_iter=1000)
            elif name == "Random Forest":
                n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
                return RandomForestClassifier(n_estimators=n_estimators)
            elif name == "SVM":
                c = st.sidebar.slider("C (Penalty)", 0.01, 10.0, 1.0)
                return SVC(C=c, probability=True)
            elif name == "Naive Bayes":
                return GaussianNB()
            elif name == "KNN":
                k = st.sidebar.slider("Number of Neighbors (K)", 1, 20, 5)
                return KNeighborsClassifier(n_neighbors=k)
            elif name == "Linear Regression":
                return LinearRegression()
            elif name == "Random Forest Regressor":
                n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
                return RandomForestRegressor(n_estimators=n_estimators)

        model = get_model(model_name)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Determine if classification or regression
        if model_name in ["Linear Regression", "Random Forest Regressor"]:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.subheader("Model Performance (Regression)")
            st.write(f"**Mean Squared Error:** {mse:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)
        else:
            acc = accuracy_score(y_test, y_pred)
            st.subheader("Model Performance (Classification)")
            st.write(f"**Accuracy:** {acc:.2f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            # ROC Curve (supports binary and multi-class)
            st.subheader("ROC Curve")
            y_test_array = y_test.to_numpy()
            class_labels = np.unique(y_test_array)

            if len(class_labels) == 2:
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig)
            else:
                y_test_bin = label_binarize(y_test, classes=class_labels)
                y_score = model.predict_proba(X_test)
                fig, ax = plt.subplots()
                for i in range(len(class_labels)):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f"Class {class_labels[i]} (AUC = {roc_auc:.2f})")
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Multiclass ROC Curve")
                ax.legend()
                st.pyplot(fig)

    else:
        st.warning("Please select feature columns.")
else:
    st.info("Awaiting CSV file upload.")
