# Import Library.
import pandas as pd

# Load Dataset.
""" https://www.kaggle.com/datasets/d4rklucif3r/social-network-ads """
dataset = pd.read_csv(r"C:\Users\DIMPU\Documents\VSCode\Social_Network_Ads.csv")
print(dataset.head())

# Split the dataset into features and target values.
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling.
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X = sc.fit_transform(X)

# Split the dataset into training and test set.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Logistic Regression Model.
""" https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression """
""" https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression """
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(random_state=42).fit(X_train, y_train)

# Train the SVM Model.
""" https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC """
""" https://scikit-learn.org/stable/modules/svm.html#svm-classification """
from sklearn.svm import SVC

svm_clf = SVC(kernel="rbf", random_state=42).fit(X_train, y_train)

# Train the Random Forest Model.
""" https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier """
""" https://scikit-learn.org/stable/modules/ensemble.html#forest """
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=42).fit(X_train, y_train)

# Train the Naive Bayes Model.
""" https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB """
""" https://scikit-learn.org/stable/modules/naive_bayes.html# """
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB().fit(X_train, y_train)

# Train the XGBoost Model.
""" XGBoost Algorithm: "https://xgboost.readthedocs.io/en/latest/" """
from xgboost import XGBClassifier

xgb = XGBClassifier().fit(X_train, y_train)

# Predict the Test set Results.
y_pred = xgb.predict(X_test)

# Predict the New Results.
print("Prediction of New Result is ", xgb.predict(sc.transform([[30, 87000]])))

# Predictions and Evaluations.
from sklearn.metrics import classification_report, confusion_matrix

print("Confusion Matrix is \n", confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
