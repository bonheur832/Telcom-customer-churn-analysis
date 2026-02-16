# IBM Telco Customer Churn Analysis  
End-to-End Data Science | Predictive Modeling | Business Analytics  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Project Status](https://img.shields.io/badge/Status-Complete-success)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-84%25-brightgreen)

---

## Executive Summary

This project presents a complete end-to-end data science workflow for analyzing and predicting customer churn using the IBM Telco Customer Churn dataset.

The objective is to identify churn drivers, understand customer behavior patterns, and build a predictive modeling framework that supports data-driven retention strategies.

This repository demonstrates practical expertise in:

- Data Cleaning and Preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Supervised Machine Learning  
- Model Evaluation  
- Business Insight Generation  

The project follows structured, production-style organization suitable for recruiter review and portfolio presentation.

---

## Business Problem

Customer churn significantly impacts revenue in the telecommunications industry. Retaining customers is more cost-effective than acquiring new ones.

This analysis addresses the following business questions:

- Which customer attributes are most strongly associated with churn?
- How do contract type, tenure, and billing structure affect churn probability?
- Which services increase customer retention?
- Can high-risk customers be identified early through predictive modeling?

---

## Dataset Information

Source: IBM Sample Dataset (Kaggle)  
File: `Telco_customer_churn.csv`  
Observations: 7,043 customers  
Features: 33 variables  
Target Variable: `Churn` (Yes / No)

The dataset includes:

- Customer demographics  
- Account information  
- Subscription services  
- Billing details  
- Payment methods  

---

## Project Structure

```
telco-customer-churn-analysis/
│
├── data/
│   └── Telco_customer_churn.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_predictive_modeling.ipynb
│
├── reports_2/
│   ├── data_exploration_insights.txt
│   ├── churn_analysis_summary.txt
│   └── figures/
│
├── main.py
├── REPORT.md
└── README.md
```

The structure supports modular development, reproducibility, and scalability.

---

## Technical Stack

Programming Language  
- Python 3.8+

Data Analysis & Processing  
- Pandas  
- NumPy  

Data Visualization  
- Matplotlib  
- Seaborn  

Machine Learning  
- Scikit-learn  

Development Environment  
- Jupyter Notebook  

---

## Methodology

### 1. Data Inspection and Validation
- Schema verification  
- Data type consistency  
- Missing value analysis  

### 2. Data Cleaning and Preprocessing
- Conversion of `TotalCharges` to numeric  
- Handling missing values  
- Encoding categorical variables  
- Feature preparation for modeling  

### 3. Exploratory Data Analysis
- Univariate and bivariate analysis  
- Churn distribution analysis  
- Correlation analysis  

### 4. Feature-Level Churn Investigation
- Contract Type  
- Tenure  
- Monthly Charges  
- Service Subscriptions  
- Payment Method  

### 5. Predictive Modeling
- Train-test split  
- Model training and evaluation  
- Accuracy and performance metrics  
- Interpretation of churn risk  

---

## Model Performance

Model Type: Supervised Classification  
Library: Scikit-learn  
Accuracy: 84%

Further evaluation details, visualizations, and interpretation are available in `REPORT.md`.

---

## Key Insights

- Month-to-month contracts show the highest churn rate.  
- Customers with short tenure (less than 12 months) are more likely to churn.  
- Higher monthly charges correlate with increased churn probability.  
- Long-term contracts significantly reduce churn risk.  
- Electronic check payment method shows elevated churn behavior.  

---

## Business Recommendations

- Incentivize long-term contracts  
- Develop onboarding engagement programs for new customers  
- Monitor high monthly charge customers for churn signals  
- Implement churn risk monitoring dashboards  

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/bonheur832/telco-customer-churn-analysis.git
cd telco-customer-churn-analysis
```

Run the main script:

```bash
python main.py
```

Open the notebooks in Jupyter:

- 01_data_exploration.ipynb  
- 02_eda_visualization.ipynb  
- 03_predictive_modeling.ipynb  

Generated outputs are available in the `reports_2/` directory.

---

## Demonstrated Skills

- Data Wrangling  
- Statistical Analysis  
- Feature Engineering  
- Predictive Modeling  
- Model Evaluation  
- Data Visualization  
- Business Insight Communication  
- Reproducible Research Practices  

---

## Author

Louis Bonheur  
Data Scientist | Machine Learning Engineer | Applied AI Researcher  

Focus Areas:  
Predictive Analytics  
Customer Behavior Modeling  
Applied Machine Learning  
AI-Driven Business Intelligence  

---