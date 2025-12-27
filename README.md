# Early Detection of Treatment Non-Response: A Longitudinal Clinical Biomarker Analysis

## 🏥 Project Overview
In many clinical treatments (e.g., immunotherapy), waiting 3-6 months to assess efficacy can be detrimental to non-responders. This project builds a machine learning system to predict **treatment failure vs. response** using only the first **5 days** of biomarker and vital sign data.

**Goal:** Provide an early warning system (Recall-optimized) to alert clinicians to potential non-responders, enabling earlier intervention (e.g., dose adjustment).

## 🛠️ Methods & Approach
### 1. Mechanistic Data Simulation
We simulated a cohort of 1,000 patients with a specific focus on **pharmacokinetic realism**:
* **Dose-Response Logic:** The simulation embeds a causal link where patients with lower **Dose-per-Kg** (e.g., high weight, fixed dose) are statistically more likely to fail treatment.
* **Log-Normal Distributions:** Used for baseline biomarker levels to mimic biological variability.
* **Longitudinal Noise:** Injected Gaussian noise and sensor errors (e.g., impossible negative values) to simulate real-world EHR messiness.

### 2. Feature Engineering
Raw time-series data was transformed into mechanistic features:
* **Dose Intensity (mg/kg):** Calculated from patient weight and fixed protocol dose to identify under-dosing risks.
* **Early Slope (Day 0-2):** Immediate physiological reaction.
* **Late Slope (Day 3-5):** Sustained effect.
* **Volatility:** Standard deviation of readings to detect unstable patients.

### 3. Model & Validation
* **Model:** Random Forest Classifier (Class-weighted).
* **Validation Strategy:** * Stratified Train/Test Split (80/20).
    * **Calibration Curve:** Verified that predicted probabilities align with observed risk.
    * **SHAP Analysis:** Used to explain *why* a specific patient was flagged (e.g., "Low Dose Intensity" driving the risk score).

## 📊 Results
* **Recall (Sensitivity) for Non-Responders:** ~92% (Key metric: minimizing False Negatives).
* **AUC-ROC:** 0.89
* **Clinical Insight:** SHAP analysis revealed that **Dose_Per_Kg** is a top predictor alongside **Biomarker Trajectory**, successfully identifying patients who failed simply because they were under-dosed for their size.

## 🚀 How to Run
1.  **Install Dependencies:** `pip install pandas scikit-learn shap fastapi uvicorn`
2.  **Run Analysis:** Execute `notebook_analysis.ipynb` to train the model and generate artifacts.
3.  **Start API:** `uvicorn app:app --reload`
4.  **Test Endpoint:** Send a POST request to `http://127.0.0.1:8000/predict` with patient vitals and **weight**.

## ⚠️ Limitations
* **Synthetic Data:** While statistically realistic, this dataset lacks complex biological confounders (genetics, drug interactions).
* **Retrospective:** This tool requires prospective clinical trial validation before use in patient care.
