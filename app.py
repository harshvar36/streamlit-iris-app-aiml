import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

# ---------------------------
# Load data
# ---------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target).map({0: "setosa", 1: "versicolor", 2: "virginica"})

# Train/test split (only for showing test accuracy in the UI)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Load a saved Decision Tree if available; otherwise train one quickly
# ---------------------------
def get_decision_tree():
    try:
        return joblib.load("iris_decision_tree_model.joblib")
    except Exception:
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        return clf

# Always have a KNN model as an option (lightweight to train)
def get_knn(k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Model & Settings")

model_choice = st.sidebar.radio(
    "Choose model:",
    ["Decision Tree", "KNN"],
    index=0
)

if model_choice == "KNN":
    k = st.sidebar.slider("K (neighbors)", min_value=1, max_value=15, value=3, step=1)
else:
    k = None

show_feature_importance = st.sidebar.checkbox("Show feature importance (Tree only)", value=True)

st.title("🌸 Iris Flower Classifier")
st.caption("Enter flower measurements on the left and get a species prediction.")

st.subheader("🧮 Input measurements")
col1, col2 = st.columns(2)

# Use dataset min/max for nice slider bounds
mins = X.min()
maxs = X.max()
means = X.mean()

with col1:
    sepal_length = st.slider("Sepal length (cm)", float(mins[0]), float(maxs[0]), float(means[0]), 0.1)
    petal_length = st.slider("Petal length (cm)", float(mins[2]), float(maxs[2]), float(means[2]), 0.1)

with col2:
    sepal_width = st.slider("Sepal width (cm)", float(mins[1]), float(maxs[1]), float(means[1]), 0.1)
    petal_width = st.slider("Petal width (cm)", float(mins[3]), float(maxs[3]), float(means[3]), 0.1)

sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# ---------------------------
# Build / load model
# ---------------------------
if model_choice == "Decision Tree":
    model = get_decision_tree()
else:
    model = get_knn(k=k)

# Evaluate on the held-out test split (just for an indicative score)
y_pred_test = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_test)

# ---------------------------
# Predict
# ---------------------------
st.markdown("---")
st.subheader("🔮 Prediction")
if st.button("Predict species"):
    pred = model.predict(sample)[0]
    st.success(f"**Predicted species:** {pred}")

    # Probabilities (if classifier supports it)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(sample)[0]
        class_names = model.classes_ if hasattr(model, "classes_") else np.unique(y)
        prob_df = pd.DataFrame({"species": class_names, "probability": probs})
        prob_df = prob_df.sort_values("probability", ascending=False).reset_index(drop=True)
        st.write("**Class probabilities:**")
        st.dataframe(prob_df, use_container_width=True)

st.caption(f"📌 Test accuracy on a small hold-out split: **{test_acc:.3f}**")

# ---------------------------
# Optional: Feature importance for Decision Tree
# ---------------------------
if model_choice == "Decision Tree" and show_feature_importance and hasattr(model, "feature_importances_"):
    st.markdown("---")
    st.subheader("🌿 Feature importance (Decision Tree)")
    importances = model.feature_importances_
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(X.columns, rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importances")
    st.pyplot(fig)

st.markdown("---")
st.caption("Created by Harshvardhan Singh")
