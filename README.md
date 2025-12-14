# 👨‍💻 ML Project: Employee CTC Prediction using Regression Models

## 🚀 Project Overview

This project focuses on building a machine learning model to accurately predict the **Cost to Company (CTC)** for newly hired employees at TechWorks Consulting. The goal is to automate and streamline the salary determination process by analyzing various factors in the provided dataset.

The core task is a **regression problem**, as the target variable (CTC) is continuous.

## 📊 Key Factors for Prediction

The salary prediction model utilizes the following features from the dataset:

* **College Tier:** Categorized tier (Tier 1, Tier 2, Tier 3) of the college attended.
* **City Type:** Categorized type of city (Metro or Non-Metro).
* **Previous CTC:** The employee's previous salary.
* **Previous job change:** Number of times the employee has changed jobs.
* **Graduation Marks:** Percentage or score in graduation.
* **EXP (Month):** Total professional experience in months.
* **Role:** The role offered in the company (Executive or Manager).

## 🛠️ Data Preprocessing & Feature Engineering

1.  **Data Loading:** Loaded the primary dataset (`ML case Study.csv`) along with auxiliary datasets for college tiers (`Colleges.csv`) and city types (`cities.csv`).
2.  **Handling Missing Values & Outliers:** Checked for missing values and duplicates (46 duplicates removed) and reviewed outliers via box plots. No significant missing values or extreme outliers were found in the final dataset.
3.  **Categorical Encoding:**
    * **College & City:** Mapped college names and city names to numerical/categorical tiers (e.g., College: 1, 2, 3; City: 1 for Metro, 0 for Non-Metro) using lists created from the auxiliary datasets.
    * **Role:** Used `pd.get_dummies()` to create one-hot encoded columns (`Role_Executive` and `Role_Manager`).

## 📈 Exploratory Data Analysis (EDA)

* **Correlation Analysis:** A correlation heatmap was generated to inspect the relationships between the independent variables and the target variable (`CTC`).
    * **Strongest positive correlations** with `CTC` were observed with `EXP (Month)` (0.296) and `Previous CTC` (0.263).
    * **Strongest negative correlation** with `CTC` was with `Role_Executive` (-0.623), which is expected as `Role_Manager` has a strong positive correlation (0.623) due to the one-hot encoding structure.

## 🤖 Machine Learning Models & Results

The dataset was split into training and testing sets (80% train, 20% test, `random_state=42`). Various regression models were trained and evaluated using **Mean Squared Error (MSE)** and the **R-Squared ($R^2$) score**.

| Model | MSE (Test) | R-Squared (R^2) Score |
| :--- | :--- | :--- |
| **Linear Regression** (Single Model) | ~73,765,335 | 0.5340 |
| **Ridge Regression** (Tuned \alpha) | ~79,774,472 | 0.5325 |
| **Lasso Regression** (Tuned \alpha) | ~79,772,449 | 0.5325 |
| **Decision Tree Regressor** (\text{max\_depth}=3) | ~67,929,108 | 0.6020 |
| **Gradient Boosting Regressor** | ~63,531,782 | 0.6277 |
| **Random Forest Regressor** | **~61,924,754** | **0.6371** |

### 🏆 Best Performing Model

The **Random Forest Regressor** achieved the best performance with an **R-Squared score of 0.6371** and the lowest Mean Squared Error on the test set.

**Reason for superior performance:** Ensemble methods like Random Forest (which uses an average of multiple decision trees) are generally more robust and effective at capturing non-linear relationships and complexities in the data compared to simpler linear models or a single Decision Tree.

## 💡 Future Steps for Improvement

To further enhance the model's predictive accuracy, the following steps are recommended:

* **Hyperparameter Tuning:** Use techniques like `GridSearchCV` or `RandomizedSearchCV` to systematically find the optimal hyperparameters for the Random Forest model.
* **Feature Engineering:** Explore interaction terms (e.g., `EXP (Month)` * `Role_Manager`) or polynomial features.
* **Ensemble Stacking:** Experiment with combining the best-performing models (like Random Forest and Gradient Boosting) using stacking or weighted averaging.
* **Outlier Treatment:** While basic checks were done, a more in-depth statistical analysis and treatment (e.g., capping/winsorizing) of outliers could be performed, particularly for `Previous CTC` and `CTC` as shown in the box plots.
