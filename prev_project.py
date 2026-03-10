import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- Selected Columns -----------------
cols = [
    'Bwd Packet Length Std',
    'PSH Flag Count',
    'min_seg_size_forward',
    'Min Packet Length',
    'ACK Flag Count',
    'URG Flag Count',
    'Init_Win_bytes_forward',
    'Bwd Packets/s',
    'Flow IAT Max',
    'Fwd IAT Std',
    'Bwd IAT Total',
    'Label'
]

# ----------------- Load Data -----------------
files = [
    "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
]

dfs = []

for file in files:
    print(f"Loading: {file}")
    
    df_temp = pd.read_csv(
        file,
        nrows=50000,      # increase later if system allows
        low_memory=False
    )
    
    df_temp.columns = df_temp.columns.str.strip()
    df_temp = df_temp.loc[:, df_temp.columns.intersection(cols)]
    dfs.append(df_temp)

df = pd.concat(dfs, ignore_index=True)
data = df.copy()

# ----------------- Label Encoding -----------------
data['Label'] = data['Label'].astype('category')
data['Label'] = data['Label'].cat.codes

y = data['Label']
X = data.drop('Label', axis=1)

# ----------------- Undersampling -----------------
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
X_rus, y_rus = rus.fit_resample(X, y)

# ----------------- SMOTE -----------------
from imblearn.over_sampling import SMOTE

sm = SMOTE(k_neighbors=1, random_state=42)
X_sm, y_sm = sm.fit_resample(X_rus, y_rus)

# ----------------- Train Test Split -----------------
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X_sm,
    y_sm,
    test_size=0.30,
    random_state=42
)

# ----------------- Scaling -----------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------- Models -----------------
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

models = [
    ('Random Forest', RandomForestClassifier(max_depth=40)),
    ('Decision Tree', tree.DecisionTreeClassifier(max_depth=33, random_state=20)),
    ('Linear SVM', LinearSVC()),
    ('KNN', KNeighborsClassifier()),
    ('Logistic Regression', LogisticRegression(solver='saga', max_iter=1000, random_state=42)),
    ('Naive Bayes', BernoulliNB())
]

# ----------------- Model Training & Evaluation -----------------
for name, model in models:
    
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(Y_test, y_pred)
    confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)
    classification = metrics.classification_report(Y_test, y_pred)
    
    print("\n==============================")
    print("Model:", name)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix)
    print("Classification Report:\n", classification)


# ----------------- Save Best Model -----------------
# Train final best model on scaled balanced data
final_model = RandomForestClassifier(max_depth=40)
final_model.fit(X_train, Y_train)

import joblib
joblib.dump(final_model, "IDS_Model.pkl")
joblib.dump(scaler, "Scaler.pkl")

print("Model and scaler saved correctly.")