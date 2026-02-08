# IBM Telco Customer Churn Analysis

## Overview
This project analyzes customer churn behavior in the telecommunications sector using the **IBM Telco Customer Churn dataset**.  
The goal is to understand **why customers leave**, identify **key churn drivers**, and provide **data-driven insights** that can support retention strategies.

This repository is designed to meet **recruiter expectations** and **Kaggle-style project standards**, with clear structure, reproducible analysis, and interpretable results.

---

## Business Problem
Customer churn is costly for telecom companies. Retaining existing customers is significantly cheaper than acquiring new ones.

**Key questions addressed:**
- Which customer characteristics are most associated with churn?
- How do contract type, tenure, and monthly charges affect churn?
- What patterns can help predict high-risk customers?

---

## Dataset
- **Source:** IBM Sample Dataset (Kaggle)
- **File:** `Telco_customer_churn.csv`
- **Rows:** 7,043 customers
- **Columns:** 33 features
- **Target Variable:** `Churn`

---

## Project Structure




---

## Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Key Analysis Steps
1. Data loading and inspection  
2. Data cleaning and preprocessing  
3. Exploratory Data Analysis (EDA)  
4. Churn behavior analysis by:
   - Contract type
   - Tenure
   - Monthly charges
   - Service usage
5. Visualization of key findings

---

## Key Insights (Summary)
- Month-to-month contracts show the highest churn rates.
- Customers with short tenure are significantly more likely to churn.
- Higher monthly charges are associated with increased churn risk.
- Long-term contracts strongly reduce churn probability.

Detailed explanations and visualizations are available in **REPORT.md**.

---

## How to Run the Project

### Requirements
- Python 3.8+
- Git
- Jupyter Notebook
1. Clone the repository
   ```bash
   git clone https://github.com/bonheur832/telco-customer-churn-analysis.git
   cd Telcom-customer-churn-analysis
   pip install -r requirements.txt
2. Navigate to the project directory
3. Run python main.py
4. Open & Run 01_data_exploration.ipynb in Jupyter Notebook
5. Open & Run 02_eda_visualization.ipynb in Jupyter Notebook
6. Open & Run 03_predictive_modeling.ipynb in Jupyter Notebook
7. Find the results in reports_2

------
Author

Louis Bonheur

Focus: Data Analysis, Machine Learning, Applied AI