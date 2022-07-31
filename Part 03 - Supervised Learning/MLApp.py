# Import Library.
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


st.title("Machine Learning Web App using Streamlit")

st.write(
    """### Explore different Classifiers and Datasets.
    Choose the Best Model?"""
)

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("KNN", "SVM", "Random Forest")
)

# Load Dataset from Sklearn Library.
def get_dataset(name):
    data = None
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)
st.write("Shape of Dataset:", X.shape)
st.write("Number of Classes:", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "KNN":
        K = st.sidebar.slider("K", 3, 15, step=2)
        params["K"] = K
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        params["max_depth"] = max_depth
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators
    return params


params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    else:
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=10,
        )
    return clf


clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=10
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy Score =", accuracy_score(y_test, y_pred))

#### PLOT DATASET ####
# Project the data onto the 2 primary principal components.
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)

# """ Run this on Terminal: `streamlit run MLApp.py` """
