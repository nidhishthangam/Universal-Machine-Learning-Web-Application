import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Clustering
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Universal ML App", layout="wide")
st.title("Universal Machine Learning App (With Clustering)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    ml_type = st.selectbox(
        "Select ML Type",
        ["Classification", "Regression", "Clustering"]
    )

    # --------------------------------------------------
    # Preprocessing (Common)
    # --------------------------------------------------
    data = df.copy()

    for col in data.select_dtypes(include="object").columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # --------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------
    if ml_type == "Classification":
        target = st.selectbox("Select Target Column", df.columns)

        X = data.drop(columns=[target])
        y = data[target]

        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_name = st.selectbox(
            "Select Algorithm",
            ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Random Forest", "Decision Tree"]
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
        }

        if st.button("Train Model"):
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {acc:.4f}")

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

    # --------------------------------------------------
    # REGRESSION
    # --------------------------------------------------
    elif ml_type == "Regression":
        target = st.selectbox("Select Target Column", df.columns)

        X = data.drop(columns=[target])
        y = data[target]

        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model_name = st.selectbox(
            "Select Algorithm",
            ["Linear Regression", "Ridge", "Lasso", "Random Forest Regressor", "Decision Tree Regressor"]
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
        }

        if st.button("Train Model"):
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
            st.success(f"R² Score: {r2_score(y_test, y_pred):.4f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

    # --------------------------------------------------
    # CLUSTERING
    # --------------------------------------------------
    else:
        cluster_algo = st.selectbox(
            "Select Clustering Algorithm",
            ["K-Means", "Hierarchical"]
        )
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)

        if st.button("Run Clustering"):
            if cluster_algo == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42)
                labels = model.fit_predict(data_scaled)

            else:
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(data_scaled)

                # Dendrogram
                st.subheader("Dendrogram")
                linked = linkage(data_scaled, method="ward")
                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(linked, ax=ax)
                st.pyplot(fig)

            df["Cluster"] = labels

            st.subheader("Clustered Data")
            st.dataframe(df.head())

            # 2D Scatter Plot
            fig, ax = plt.subplots()
            ax.scatter(
                data_scaled[:, 0],
                data_scaled[:, 1],
                c=labels
            )
            ax.set_title("Cluster Visualization (First 2 Features)")
            st.pyplot(fig)
