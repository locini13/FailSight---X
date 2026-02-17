SightX — Predictive Truck Failure System Using Survival Analysis

Project Overview

SightX is a predictive maintenance system designed to estimate truck failure risk and remaining operational life using survival analysis techniques.
Heavy-duty trucks generate large volumes of sensor and operational data during daily usage. However, traditional machine learning approaches are not well suited for failure prediction in this domain because many trucks remain operational throughout the observation period. This creates censored data, where the exact failure time is unknown.
To address this limitation, this project applies survival modeling to predict both time-to-failure and failure probability, enabling proactive maintenance decisions and reducing unexpected breakdowns.

Problem Statement

Unexpected truck failures can result in operational disruption, increased repair costs, and safety concerns. The primary challenge is not only determining whether a truck will fail, but also estimating when the failure is likely to occur.
Since a large portion of the fleet has not yet failed, conventional classification models cannot effectively utilize the available data.

Objective:
Develop a predictive system that estimates truck survival time, identifies high-risk vehicles, explains model predictions, and quantifies the financial benefits of preventive maintenance.

Solution Approach

The project implements an end-to-end machine learning pipeline structured across multiple stages.

1. Data Preparation

Merged operational readouts, repair records, and vehicle specifications.
Handled missing values using backfilling techniques.
Engineered Remaining Useful Life (RUL) as the survival target.
Removed low-correlation features and eliminated multicollinearity.
Produced a clean dataset suitable for survival modeling.

2. Survival Modeling

Multiple survival models were trained to learn time-to-failure patterns:
Cox Proportional Hazards (baseline)
Random Survival Forest (best performing model)
Gradient Boosting Survival Analysis
Each model uses two key variables:
Time: Remaining Useful Life
Event: Whether a failure occurred
Model performance was evaluated using the Concordance Index (C-index) to measure predictive accuracy.

3. Risk Prediction
   
Survival probabilities were computed at a future time horizon of 50 operational cycles to answer the question:
“Will the truck survive the next 50 cycles?”
Based on these probabilities, trucks were categorized into risk groups
Survival Probability	Risk Category
Greater than 0.80	Low Risk
Between 0.60 and 0.80	Medium Risk
Less than 0.60	High Risk
This enables maintenance teams to intervene before failures occur.

4. Model Explainability

SHAP (SHapley Additive Explanations) was implemented to interpret model predictions.
This provides visibility into:
Which sensor readings influence failure risk
Why a specific truck is classified as high risk
How individual features push predictions higher or lower
Explainability improves model transparency and supports data-driven operational decisions.

5. Cost-Benefit Analysis

A maintenance cost simulation was developed to compare reactive and predictive strategies.
The analysis demonstrated that early intervention based on survival predictions can significantly reduce overall maintenance expenses by preventing unexpected failures.

Technology Stack
Python
Pandas, NumPy
Scikit-Learn
Scikit-Survival
SHAP
Matplotlib, Seaborn
Joblib
Streamlit 

## Project Structure

```
scania_rul_survival/
│
├── data/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_survival_modeling.ipynb
│   ├── 03_risk_prediction.ipynb
│   ├── 04_model_explainability.ipynb
│   └── 05_cost_analysis.ipynb
│
├── models/        
├── results/       
├── app.py
└── README.md
```


Key Contributions

Built a complete predictive maintenance pipeline from raw telemetry data.
Addressed censored data challenges using survival analysis.
Predicted failure risk before breakdown events.
Integrated explainable AI to improve interpretability.
Demonstrated measurable financial impact through cost simulation.
Business Impact
The system supports a transition from reactive maintenance to predictive maintenance by:
Reducing unexpected truck failures
Optimizing maintenance scheduling
Improving fleet reliability
Lowering operational costs

Future Enhancements
Real-time sensor data ingestion
Automated model retraining pipeline
Experiment tracking with MLflow
Advanced cost optimization
Fleet monitoring dashboard
