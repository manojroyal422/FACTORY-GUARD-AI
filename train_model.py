# ==========================================
# FactoryGuard AI - Production Version
# Predictive Maintenance System
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score
)

from xgboost import XGBClassifier

sns.set(style="whitegrid")

# ==========================================
# 1️⃣ LOAD DATA
# ==========================================

data_path = r"C:\Users\manoj\Downloads\ai4i2020.csv"
df = pd.read_csv(data_path)

# Remove leakage columns
df = df.drop(['UDI','Product ID'], axis=1)

# ==========================================
# 2️⃣ FEATURE ENGINEERING
# ==========================================

df['Torque_Wear'] = df['Torque [Nm]'] * df['Tool wear [min]']
df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Speed_Torque'] = df['Rotational speed [rpm]'] * df['Torque [Nm]']

# Encode categorical
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

# Clean column names
df.columns = df.columns.str.replace('[','', regex=False)
df.columns = df.columns.str.replace(']','', regex=False)
df.columns = df.columns.str.replace(' ','_')

# ==========================================
# 3️⃣ FEATURES & TARGET
# ==========================================

X = df.drop('Machine_failure', axis=1)
y = df['Machine_failure']

# ==========================================
# 4️⃣ DATA SPLIT
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.4,
    stratify=y,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=42
)

# ==========================================
# 5️⃣ HANDLE CLASS IMBALANCE
# ==========================================

ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])

# ==========================================
# 6️⃣ OPTUNA HYPERPARAMETER TUNING
# ==========================================

def objective(trial):

    params = {

        "objective":"binary:logistic",
        "eval_metric":"logloss",

        "max_depth": trial.suggest_int("max_depth",3,10),
        "learning_rate": trial.suggest_float("learning_rate",0.01,0.2),
        "n_estimators": trial.suggest_int("n_estimators",200,700),

        "subsample": trial.suggest_float("subsample",0.6,1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree",0.6,1.0),

        "gamma": trial.suggest_float("gamma",0,5),

        "scale_pos_weight": ratio,

        "tree_method":"hist",
        "random_state":42
    }

    model = XGBClassifier(**params)

    model.fit(X_train,y_train)

    preds = model.predict_proba(X_val)[:,1]

    return roc_auc_score(y_val,preds)


study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=30)

best_params = study.best_params

best_params["scale_pos_weight"] = ratio
best_params["objective"] = "binary:logistic"
best_params["eval_metric"] = "logloss"

print("\nBest Hyperparameters:")
print(best_params)

# ==========================================
# 7️⃣ TRAIN FINAL MODEL
# ==========================================

model = XGBClassifier(**best_params)

model.fit(X_train,y_train)

# Save model to FactoryGuardAI folder
joblib.dump(model, r"D:\FactoryGuardAI\factoryguard_model.pkl")

print("\nModel saved to D:\\FactoryGuardAI")

# ==========================================
# 8️⃣ THRESHOLD OPTIMIZATION
# ==========================================

val_probs = model.predict_proba(X_val)[:,1]

best_threshold = 0.5
best_f1 = 0

for t in np.arange(0.1,0.9,0.01):

    preds = (val_probs>t).astype(int)

    score = f1_score(y_val,preds)

    if score>best_f1:

        best_f1 = score
        best_threshold = t

print("\nBest Threshold:",round(best_threshold,3))

# Save threshold
joblib.dump(best_threshold, r"D:\FactoryGuardAI\threshold.pkl")

print("Threshold saved to D:\\FactoryGuardAI")

# ==========================================
# 9️⃣ FINAL TEST EVALUATION
# ==========================================

test_probs = model.predict_proba(X_test)[:,1]

y_pred = (test_probs>best_threshold).astype(int)

print("\n===== FINAL MODEL PERFORMANCE =====")

print(classification_report(y_test,y_pred))

print("ROC AUC:",roc_auc_score(y_test,test_probs))

# ==========================================
# 🔟 CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Failure","Failure"],
    yticklabels=["No Failure","Failure"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================
# 1️⃣1️⃣ ROC CURVE
# ==========================================

fpr,tpr,_ = roc_curve(y_test,test_probs)

plt.figure(figsize=(6,4))

plt.plot(fpr,tpr,label="Model")
plt.plot([0,1],[0,1],'--')

plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()

plt.show()

# ==========================================
# 1️⃣2️⃣ FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

imp_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":importance
}).sort_values(by="Importance",ascending=False)

plt.figure(figsize=(8,6))

sns.barplot(x="Importance",y="Feature",data=imp_df.head(10))

plt.title("Top Factors Causing Machine Failure")

plt.show()

# ==========================================
# 1️⃣3️⃣ RISK REPORT
# ==========================================

results = X_test.copy()

results["Failure_Probability"] = test_probs
results["Predicted_Failure"] = y_pred

def risk_level(p):

    if p>0.7:
        return "HIGH RISK"

    elif p>0.4:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


results["Risk_Level"] = results["Failure_Probability"].apply(risk_level)

results_sorted = results.sort_values(
    by="Failure_Probability",
    ascending=False
)

print("\n===== TOP 10 MACHINES REQUIRING MAINTENANCE =====")

print(results_sorted[["Failure_Probability","Risk_Level"]].head(10))

# ==========================================
# 1️⃣4️⃣ SHAP EXPLAINABILITY
# ==========================================

explainer = shap.Explainer(model)

shap_values = explainer(X_test)

# Summary plot
shap.plots.beeswarm(shap_values)

# Explanation for highest risk machine

top_index = results_sorted.index[0]

top_position = X_test.index.get_loc(top_index)

print("\n===== EXPLANATION FOR HIGHEST RISK MACHINE =====")

print(
    "Failure Probability:",
    round(results.loc[top_index,"Failure_Probability"],3)
)

print(
    "Risk Level:",
    results.loc[top_index,"Risk_Level"]
)

shap.plots.waterfall(shap_values[top_position])