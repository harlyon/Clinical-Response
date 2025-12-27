import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, auc
)
from sklearn.calibration import calibration_curve

# ========== 1. Configuration ==========
np.random.seed(42)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ========== 2. Data Loading ==========
df = pd.read_csv("clinical_data.csv")
print("Data loaded. Shape:", df.shape)

# ========== 3. Data Cleaning & Feature Engineering ==========
def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Handle sensor errors (negative or extreme values)
    biomarker_cols = [c for c in df.columns if 'Biomarker' in c]
    for col in biomarker_cols:
        df[col] = df[col].mask((df[col] < 0) | (df[col] > 1000), np.nan)
    # Linear interpolation for time-series
    df[biomarker_cols] = df[biomarker_cols].interpolate(method='linear', axis=1, limit_direction='both')
    # Trajectory features
    df['Total_Change'] = df['Biomarker_Day5'] - df['Biomarker_Day0']
    df['Pct_Change'] = (df['Biomarker_Day5'] - df['Biomarker_Day0']) / (df['Biomarker_Day0'] + 1e-6)
    df['Early_Change'] = df['Biomarker_Day2'] - df['Biomarker_Day0']
    df['Late_Change'] = df['Biomarker_Day5'] - df['Biomarker_Day3']
    df['Biomarker_Volatility'] = df[biomarker_cols].std(axis=1)
    # Outlier capping (IQR)
    cols_to_cap = ['Total_Change', 'Pct_Change', 'Biomarker_Volatility', 'Early_Change', 'Late_Change']
    for col in cols_to_cap:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        if 'Volatility' in col:
            low = max(0, low)
        df[col] = df[col].clip(lower=low, upper=high)
    return df

df = clean_and_engineer(df)
print("Data cleaned and features engineered.")

# ========== 4. Modeling ==========
def train_model(df: pd.DataFrame):
    # Features and target
    X = df.drop(['Patient_ID', 'Responder_Status', 'Weight_Kg'], axis=1)
    y = df['Responder_Status']
    # Encode categoricals
    X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Scaling
    scaler = StandardScaler()
    cols_to_scale = ['Age', 'Baseline_Severity', 'Biomarker_Day0', 'Dose_Per_Kg']
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    # Model
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight='balanced', random_state=42
    )
    rf_model.fit(X_train, y_train)
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/clinical_response_model.pkl")
    joblib.dump(scaler, "models/clinical_scaler.pkl")
    print("Model trained and artifacts saved.")
    return rf_model, scaler, X_train, X_test, y_train, y_test

rf_model, scaler, X_train, X_test, y_train, y_test = train_model(df)

# ========== 5. Evaluation ==========
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Responder', 'Responder']))
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    # Feature importance
    importances = model.feature_importances_
    feature_names = X_test.columns
    indices = np.argsort(importances)[::-1]
    # Plots
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    axes = axes.flatten()
    # ROC
    axes[0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[0].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Chance')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve: Model Discrimination')
    axes[0].legend(loc="lower right")
    # Calibration
    axes[1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Random Forest')
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    axes[1].set_xlabel('Predicted Probability of Response')
    axes[1].set_ylabel('Actual Fraction of Responders')
    axes[1].set_title('Calibration Curve')
    axes[1].legend()
    # Distribution by outcome
    sns.histplot(y_prob[y_test == 1], bins=20, color='green', label='Responder', kde=True, stat="density", alpha=0.6, ax=axes[2])
    sns.histplot(y_prob[y_test == 0], bins=20, color='red', label='Non-Responder', kde=True, stat="density", alpha=0.6, ax=axes[2])
    axes[2].set_xlabel('Predicted Probability of Response')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Distribution of Predicted Probabilities by Outcome')
    axes[2].legend()
    # Feature importance
    sns.barplot(x=importances[indices][:15], y=feature_names[indices][:15], palette='viridis', ax=axes[3])
    axes[3].set_title('Top 15 Feature Importances')
    axes[3].set_xlabel('Importance')
    axes[3].set_ylabel('Feature')
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=axes[4], cmap='Blues', colorbar=False)
    axes[4].set_title('Confusion Matrix')
    axes[5].set_visible(False)
    plt.tight_layout()
    plt.show()

evaluate_model(rf_model, X_test, y_test)

# ========== 6. Interpretability ==========
def interpret_model(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(vals, X_test, plot_type="dot", show=False)
    plt.gcf().set_size_inches(38, 5)
    plt.show()

interpret_model(rf_model, X_test)

if __name__ == "__main__":
    print("Pipeline complete.")