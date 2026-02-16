# **Project Overview**

Employee turnover is a major challenge for organizations, with replacement costs ranging from 50% to 200% of an employee’s annual salary. This project develops and compares multiple machine learning models to predict employee attrition using real-world HR data. By identifying employees at risk of leaving, organizations can shift from reactive resignation management to proactive retention strategies.

# **Dataset**

- 59,598 employee records
- 5 industries: Education, Media, Healthcare, Technology, Finance
- 18 original features (expanded to 21 after encoding)
- Target variable: Attrition (1 = Left, 0 = Stayed)
- Balanced classes (47.55% attrition rate)

# **Technical Approach**
## **Feature Categories**

- Demographics (Age, Gender, Marital Status, Education, Dependents)
- Job Characteristics (Job Level, Promotions, Income, Tenure)
- Work Environment (Work-Life Balance, Job Satisfaction, Reputation, Recognition)

## **Data Preprocessing**

- Ordinal encoding for ranked variables
- One-hot encoding for nominal variables
- Stratified 80/20 train-test split
- Z-score standardization (for distance-based models)
- Multicollinearity check (|r| > 0.8 threshold)
- Feature selection via Random Forest importance (reduced 21 → 10 features)Removed non-predictive Employee ID

## **Models Implemented**

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Classification and Regression Trees (CART)
- Random Forest
- Naive Bayes

# **Best Model:**
Random Forrest (100 trees, max depth=10)

# **Key Predictors**

- Job Level (27% feature importance)
- Marital Status
- Work-Life Balance
- Distance from Home
- Number of Promotions
Top 10 features captured ~90% of total predictive importance.

# **Technologies Used**
Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | Random Forest Feature Importance

# **Business Value**

- Enables proactive HR retention strategies
- Identifies ~71% of employees likely to leave
- Supports data-driven workforce planning

# **Author**

Yash Pratap Singh/ https://www.linkedin.com/in/yash-pratap-singh2202/
