# Develop and Deploy A Diabetes Prescreening Model based on Lifestyle Habit
Try the app here: [Live App](https://diabetes-prescreen.streamlit.app/)

## Motivation
- According to WHO, silent-killer diseases, including diabetes, cause [75% of all global deaths](https://www.who.int/news-room/spotlight/the-silentkillers), and nearly half of people with diabetes remain undiagnosed and without noticeable symptoms until irreversible complications like heart disease, kidney failure, or vision loss may already be underway.
- Alarmingly, the prevalence of type 2 diabetes in younger adults and adolescents is increasing. For example, recent U.S. trends show [type 2 diagnoses in those under 20 have doubled over 16 years](https://www.cdc.gov/diabetes/data-research/research/trends-new-diabetes-cases-young-people.html).
-  **Early detection and lifestyle awareness**: This app provides a quick, accessible, and non-invasive way to assess your risk of diabetes through everyday lifestyle questions, long before symptoms appear. It especially empowers the younger generation to encourage proactive health choices that can reduce or even reverse diabetes risk.

## Overview
This project is a diabetes risk prescreening tool based on lifestyle and demographic factors.
The app is built with Streamlit and powered by a Logistic Regression model, trained and validated on the CDC Diabetes Health Indicators Dataset (253,680 participants).
Instead of requiring clinical tests (e.g., blood work), the model uses 13 lifestyle and demographic questions to provide a probabilistic risk score.

## Features
- **Interactive prediction**: Input lifestyle and demographic data, get risk probability & classification
- Visualization: **Gauge chart and metrics** for interpretation
- Risk factor insights: **Personalized feedback** on lifestyle-related risk factors
- Model information page: Learn how the model was trained, evaluated, and validated
- Deployment ready: Run locally, via Docker, or on Streamlit Cloud

## Model
- See the complete model building [here](https://github.com/ignsagita/class-diabetes-lifestyle/blob/main/class-diabetes_lifestyle.ipynb)
- Algorithm: Logistic Regression (L1 regularization, C=0.01)
- *Why Logistic Regression?*
  - Interpretable (coefficients reflect risk impact)
  - Probabilistic output (calibrated risk score)
  - Outperformed Random Forest and LightGBM in PR-AUC and ROC-AUC
- Performance (5-fold Stratified CV):
  - ROC-AUC: 0.759 ± 0.001
  - PR-AUC: 0.525
  - Optimal Threshold: 0.547
- Selected Features (13):
  - Demographics: Age, Sex, Education, Income
  - Lifestyle: Smoking, Physical Activity, Fruits, Vegetables, Alcohol, Healthcare Access, Cost Barriers, General Health, Walking Difficulty

## Model Training Pipeline
1. Data Preprocessing + Feature Engineering
2. Model selection: compared logistic regression, random forest, and lightGBM
3. Feature selection: retained features only if PR-AUC >= 0.7 or ROC-AUC >= 0.8 on validation fold
4. Nested Cross-Validation: 3 stratified folds in outer loop and 5 stratified folds in inner loop (for hyperparameter tuning)

## Dataset
- Source: CDC Diabetes Health Indicators [UCI Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- Size: 253,680 participants from US
- Original Features: 35
- Final Features Used: 13 (accessible without medical testing)

## Quick Start
- Local setup
  - Clone repo: git clone https://github.com/ignsagita/class-diabetes-lifestyle.git cd class-diabetes-lifestyle
  - Install dependencies: pip install -r requirements.txt
  - Run app: streamlit run stream_app.py
  - visit http://localhost:8501
 
 - Run with Docker
   - docker build -t class-diabetes-lifestyle .
   - docker run -p 8501:8501 class-diabetes-lifestyle
   - visit http://localhost:8501

## Project Structure
class-diabetes-lifestyle/<br>
│ <br>
├── model/                  # Trained logistic regression model (.pkl) <br>
├── stream_app.py           # Main Streamlit application <br>
├── requirements.txt        # Python dependencies <br>
├── class-diabetes_lifestyle.ipynb  # Training & evaluation notebook <br>
└── README.md               # Project documentation <br>

## Disclaimer:
*This tool is for educational purposes only. It is not a diagnostic tool and should not replace professional medical consultation.
Always consult healthcare professionals for medical advice and decisions.*

---
