import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="CTG ML Playground", layout="wide")

# Upload dataset
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload cardiotocography_v2.csv file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.title("Cardiotocogram ML Analysis Playground")
    st.write("### First 5 rows of the dataset")
    st.dataframe(df.head())

    # Data Mining
    st.header("1. Data Mining - Exploratory Analysis")
    st.write("Shape of data:", df.shape)
    st.write("Column names:", list(df.columns))
    st.write("Missing values per column:")
    st.write(df.isnull().sum())
    st.write("Descriptive statistics:")
    st.write(df.describe())
    st.write("Class distribution:")
    st.write(df['CLASS'].value_counts().to_frame())

    # Data Preparation & Preprocessing
    st.header("2. Data Preparation and Preprocessing")
    feature_cols = [col for col in df.columns if col != 'CLASS']
    X = df[feature_cols]
    y = df['CLASS']

    # Imputation (Fill missing values)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=feature_cols)

    test_size = st.sidebar.slider('Validation set size', 0.1, 0.4, 0.2)
    random_state = st.sidebar.number_input("Random State / Seed", value=42)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Preprocessing choice
    st.sidebar.subheader("Preprocessing Method")
    preprocessing = st.sidebar.radio("Choose one", ['Raw', 'Standardization', 'Normalization', 'PCA (10 dim)'])

    if preprocessing == "Raw":
        X_tr, X_v = X_train.values, X_val.values
    elif preprocessing == "Standardization":
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_v = scaler.transform(X_val)
    elif preprocessing == "Normalization":
        norm = Normalizer()
        X_tr = norm.fit_transform(X_train)
        X_v = norm.transform(X_val)
    elif preprocessing == "PCA (10 dim)":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        pca = PCA(n_components=10, random_state=42)
        X_tr = pca.fit_transform(X_train_s)
        X_v = pca.transform(X_val_s)

    # Classifier selection
    st.header("3. Model Selection and Hyperparameters")
    clf_name = st.selectbox("Choose a model", ["Naive Bayes", "Decision Tree", "Random Forest", "SVM"])


    # Hyperparameter selection
    if clf_name == "Naive Bayes":
        var_smoothing = st.selectbox(
            "Choose var_smoothing",
            [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            index=1
        )
        clf = GaussianNB(var_smoothing=var_smoothing)
    elif clf_name == "Decision Tree":
        max_depth = st.slider("max_depth", 1, 40, 7)
        min_samples_leaf = st.slider("min_samples_leaf", 1, 25, 1)
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    elif clf_name == "Random Forest":
        n_estimators = st.slider("n_estimators", 10, 300, 100, 10)
        max_depth = st.slider("max_depth (RF)", 1, 40, 7)
        min_samples_leaf = st.slider("min_samples_leaf (RF)", 1, 25, 1)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
    elif clf_name == "SVM":
        kernel = st.selectbox("Kernel", ['linear', 'rbf', 'poly', 'sigmoid'])
        C = st.slider("Regularization parameter (C)", 0.01, 10.0, 1.0)
        gamma = st.selectbox("Gamma", ['scale', 'auto'])
        clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=random_state)

    # Fit model and predict
    if st.button("Train and Evaluate Model"):
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_v)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        st.success(f"Accuracy: {acc:.4f}")
        st.info(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")


        # Confusion matrix with heatmap
        st.write("### Confusion Matrix (Heatmap)")
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Show feature importances if possible
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            feat_names = [f"PC-{i+1}" for i in range(imp.shape[0])] if preprocessing.startswith("PCA") else feature_cols
            st.write("Feature importances:")
            st.dataframe(pd.DataFrame({"Feature": feat_names, "Importance": imp}).sort_values("Importance", ascending=False))
else:
    st.warning("Please upload your data file (CSV) to get started!")