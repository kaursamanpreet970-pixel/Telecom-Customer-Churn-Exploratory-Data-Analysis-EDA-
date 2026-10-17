# 📡 Telecom Customer Churn – Exploratory Data Analysis (EDA)

> **One-line summary:** A Python-based exploratory data analysis of 7,043 telecom customers to find out *who* leaves, *when* they leave, and *why* — using pandas, matplotlib and seaborn.

---

## 📑 Table of Contents

1. [Project Title](#-telecom-customer-churn--exploratory-data-analysis-eda)
2. [Brief One Line Summary](#-one-line-summary)
3. [Overview](#1-overview)
4. [Problem Statement](#2-problem-statement)
5. [Dataset](#3-dataset)
6. [Tools and Technologies](#4-tools-and-technologies)
7. [Methods](#5-methods-step-by-step)
8. [Key Insights](#6-key-insights)
9. [Dashboard / Model / Output](#7-dashboard--model--output)
10. [How to Run This Project](#8-how-to-run-this-project)
11. [Results & Conclusion](#9-results--conclusion)
12. [Connect With Me](#-connect-with-me)

---

## 1. Overview

Customer churn means a customer stops using a company's service. In the telecom industry, winning a new customer costs far more than keeping an existing one, so understanding churn is one of the most valuable analyses a telecom company can run.

This project explores a telecom customer dataset end to end:

- loading and inspecting the raw data,
- checking data quality and fixing problems,
- visualising churn across customer profiles, contracts, services and billing,
- and turning the charts into clear, business-ready insights.

It is a pure **EDA (Exploratory Data Analysis)** project — no machine learning model is built here. The goal is to understand the data and tell the story behind it.

---

## 2. Problem Statement

A telecom company is losing customers, and about **1 in 4 customers has already churned**. The business wants answers to these questions:

| # | Business question |
|---|-------------------|
| 1 | What percentage of customers are churning? |
| 2 | Do demographics (gender, senior citizen status) influence churn? |
| 3 | At what point in the customer lifetime (tenure) do people leave? |
| 4 | Does the type of contract affect churn? |
| 5 | Which services (internet type, security, tech support, backup, etc.) are linked to higher churn? |
| 6 | Does the payment method or billing style make a difference? |
| 7 | What can the company do to reduce churn? |

**Why it matters:** if the company knows the profile of a customer who is likely to leave, it can act early with targeted offers, better onboarding and better service bundles instead of reacting after the customer is gone.

---

## 3. Dataset

- **Name:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Source:** IBM sample dataset, publicly available on Kaggle
- **Size:** 7,043 rows (customers) × 21 columns (features)
- **Target variable:** `Churn` (Yes / No)

### 3.1 Raw dataset columns

| Group | Column | Description |
|-------|--------|-------------|
| Customer | `customerID` | Unique ID of each customer |
| Demographics | `gender` | Male / Female |
| | `SeniorCitizen` | Whether the customer is a senior citizen (originally 0 / 1) |
| | `Partner` | Whether the customer has a partner |
| | `Dependents` | Whether the customer has dependents |
| Account | `tenure` | Number of months the customer has stayed (0–72) |
| | `Contract` | Month-to-month, One year or Two year |
| | `PaperlessBilling` | Whether billing is paperless |
| | `PaymentMethod` | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| Services | `PhoneService` | Whether the customer has phone service |
| | `MultipleLines` | Yes / No / No phone service |
| | `InternetService` | DSL, Fiber optic or No |
| | `OnlineSecurity` | Yes / No / No internet service |
| | `OnlineBackup` | Yes / No / No internet service |
| | `DeviceProtection` | Yes / No / No internet service |
| | `TechSupport` | Yes / No / No internet service |
| | `StreamingTV` | Yes / No / No internet service |
| | `StreamingMovies` | Yes / No / No internet service |
| Charges | `MonthlyCharges` | Amount charged every month |
| | `TotalCharges` | Total amount charged over the customer's lifetime |
| Target | `Churn` | Whether the customer left the company (Yes / No) |

### 3.2 Problems found in the raw data

Real-world data is never perfectly clean. These are the issues found and how each was handled:

| # | Problem in the raw data | Why it is a problem | How it was solved |
|---|-------------------------|---------------------|-------------------|
| 1 | `TotalCharges` is stored as **text (string)** instead of a number | Numeric operations such as `describe()`, sums and averages would not work | Converted the column to `float` |
| 2 | `TotalCharges` has **blank values (" ")** for customers whose `tenure` is 0 | Blank strings stop the type conversion from working | These customers are brand new and have not been billed yet, so the blanks were replaced with `0` |
| 3 | `SeniorCitizen` is coded as **0 / 1**, while every other Yes/No column uses text | Inconsistent labels make charts hard to read | Converted 0 → `No` and 1 → `Yes` |
| 4 | **Class imbalance** in the target: 5,174 stayed vs 1,869 churned | Raw counts alone can hide patterns in small groups, and a future model could be biased towards "No" | Used **percentage (normalised) charts** where group sizes differ |
| 5 | Columns contain extra categories such as *"No internet service"* and *"No phone service"* | They are different from a plain "No" and must be read carefully | Kept them as their own category in the charts |

### 3.3 Data quality check

Run on the raw data before any cleaning:

```
missing values : 0
duplicate rows : 0
duplicate IDs  : 0
```

---

## 4. Tools and Technologies

| Tool | Purpose |
|------|---------|
| **Python 3** | Programming language |
| **Jupyter Notebook** | Writing and presenting the analysis |
| **pandas** | Loading, cleaning and summarising data |
| **NumPy** | Numerical operations |
| **Matplotlib** | Base plotting (pie chart, stacked bars, figure layout) |
| **Seaborn** | Statistical charts (countplots, histogram) |
| **math** | Calculating the subplot grid size |

---

## 5. Methods (Step by Step)

### Step 1 – Import libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
```

### Step 2 – Load and inspect the data

```python
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.shape      # (7043, 21)
df.info()     # column types and non-null counts
```

### Step 3 – Check data quality

```python
print("missing:", df.isna().sum().sum(),
      "| duplicate rows:", df.duplicated().sum(),
      "| duplicate IDs:", df.customerID.duplicated().sum())
```

### Step 4 – Clean the data

```python
# TotalCharges: blank -> 0 (tenure is 0, so nothing has been billed yet), then convert to number
df['TotalCharges'] = df["TotalCharges"].replace(" ", "0")
df['TotalCharges'] = df["TotalCharges"].astype("float")

# SeniorCitizen: 0/1 -> No/Yes
df["SeniorCitizen"] = df["SeniorCitizen"].astype("str")
df["SeniorCitizen"] = df["SeniorCitizen"].replace("0", "No")
df["SeniorCitizen"] = df["SeniorCitizen"].replace("1", "Yes")
```

### Step 5 – Statistical summary

```python
df.describe()
```

| Metric | tenure (months) | MonthlyCharges | TotalCharges |
|--------|-----------------|----------------|--------------|
| Mean | 32.4 | 64.76 | 2,279.73 |
| Min | 0 | 18.25 | 0 |
| Median | 29 | 70.35 | 1,394.55 |
| Max | 72 | 118.75 | 8,684.80 |

### Step 6 – Visualise churn (Data Visualization)

Each chart below answers one business question. See the next section for the charts and what they show.

| Chart | Seaborn / Matplotlib function used |
|-------|-----------------------------------|
| Churn count | `sns.countplot` |
| Churn rate | `plt.pie` |
| Churn by gender | `sns.countplot(hue="Churn")` |
| Senior citizens (count and %) | `sns.countplot` and `pd.crosstab(normalize='index')` + stacked bar |
| Churn by tenure | `sns.histplot(bins=72, hue="Churn")` |
| Churn by contract | `sns.countplot(hue="Churn")` |
| 12 service/billing features | `plt.subplots` grid with a loop of `sns.countplot` |
| Churn by payment method | `sns.countplot(hue="Churn")` |

---

## 6. Key Insights

### 📊 Insight 1 – Overall churn: about 1 in 4 customers leaves

Of 7,043 customers, **1,869 churned and 5,174 stayed**, a churn rate of **26.54%**. The dataset is imbalanced, with far more retained customers than churned ones.

![Churn count](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/01_churn_count.png)

![Churn rate pie chart](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/02_churn_pie.png)

---

### 👥 Insight 2 – Gender does not drive churn

Female (939) and male (930) churn counts are almost identical, at roughly 27% vs 26%. **Gender is not a meaningful factor.**

![Churn by gender](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/03_churn_by_gender.png)

---

### 👴 Insight 3 – Senior citizens churn much more

Senior citizens are a smaller group, so raw counts hide the problem. The percentage chart shows it clearly: **41.7% of senior citizens churn, compared with 23.6% of non-seniors.**

![Senior citizen churn count](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/04_senior_citizen_count.png)

![Senior citizen churn percentage](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/05_senior_citizen_percent.png)

---

### ⏳ Insight 4 – Customers leave early

Churn is heavily concentrated in the **first few months**, with the biggest churn spike at **month 1**. It falls steadily as tenure grows. Customers who stay long term, especially near 72 months, rarely leave. **The early months are the critical retention window.**

![Churn by tenure](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/06_churn_by_tenure.png)

---

### 📝 Insight 5 – Contract type is the strongest indicator

| Contract | Churned | Churn rate |
|----------|---------|-----------|
| Month-to-month | 1,655 of 3,875 | **about 43%** |
| One year | 166 of 1,473 | about 11% |
| Two year | 48 of 1,695 | under 3% |

Longer commitments clearly improve retention.

![Churn by contract](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/07_churn_by_contract.png)

---

### 🌐 Insight 6 – Internet type and add-on services matter

The grid below compares churn across 12 service and billing features. What stands out:

- **Fiber optic** customers churn far more than DSL customers or customers with no internet service.
- Customers **without Online Security, Tech Support, Online Backup or Device Protection** churn much more than those who have these add-ons. These services seem to help retention.
- **Phone service** and **multiple lines** show little difference in churn, so they are weak predictors.

![Service and billing features](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/08_service_features_grid.png)

---

### 💳 Insight 7 – Electronic check users churn the most

| Payment method | Churned | Churn rate |
|----------------|---------|-----------|
| Electronic check | 1,071 of 2,365 | **about 45%** |
| Mailed check | 308 of 1,612 | about 19% |
| Bank transfer (automatic) | 258 of 1,544 | about 17% |
| Credit card (automatic) | 232 of 1,522 | about 15% |

Customers on automatic payments seem more likely to stay.

![Churn by payment method](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/09_churn_by_payment_method.png)

---

## 7. Dashboard / Model / Output

This is an EDA project, so the output is a fully documented notebook with **9 data visualizations** and written insights next to each chart.

| Output | File |
|--------|------|
| Complete analysis notebook (code, charts and insights) | [`telecom_eda.ipynb`](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/telecom_eda.ipynb) |
| PDF export of the notebook (easy to read and share) | [`telecom_churn_eda.pdf`](Telecom-Customer-Churn-Exploratory-Data-Analysis-EDA-/blob/main/telecom_churn_eda.pdf) |


## 9. Results & Conclusion

### Result

The typical customer who churns is:

> **a new customer, on a month-to-month contract, using fiber optic internet, paying by electronic check, and without online security or tech support add-ons.**

Senior citizens are also at higher risk, while gender makes no real difference.

### Recommendations

1. **Offer incentives** to move customers from month-to-month to one- or two-year contracts.
2. **Bundle Online Security and Tech Support** for new customers, since these are linked to much lower churn.
3. **Encourage automatic payment methods** such as bank transfer or credit card.
4. **Focus onboarding and support on the first 3 months**, when customers are most likely to leave.
5. **Investigate fiber optic service quality and pricing**, given its much higher churn.

### Future work

- Build a churn prediction model (Logistic Regression, Random Forest, XGBoost) and handle the class imbalance.
- Add `MonthlyCharges` and `TotalCharges` analysis against churn.
- Build an interactive Power BI / Tableau dashboard of these findings.

---

## 🤝 Connect With Me

**Samanpreet Kaur**

| Platform | Link |
|----------|------|
| 💼 LinkedIn | [Samanpreet Kaur on LinkedIn](www.linkedin.com/in/kaursamanpreet) |
| 📧 Email | [YOUR-EMAIL@example.com](mailto:kaursamanpreet970@gmail.com) |
| 🐙 GitHub | [github.com/kaursamanpreet970-pixel](kaursamanpreet970-pixel) |

⭐ If you found this project useful, please consider giving the repository a star!
