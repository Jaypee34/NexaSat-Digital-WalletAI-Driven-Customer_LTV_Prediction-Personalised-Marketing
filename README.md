# NexaSat Digital Wallet – AI-Driven Customer Lifetime Value (CLV) Prediction

## 1. Project Overview

The **NexaSat Digital Wallet – AI-Driven Customer Lifetime Value (CLV) Prediction** project develops a machine-learning solution to estimate the lifetime value of digital-wallet customers using demographic, behavioural, transactional and engagement-related data.

The project addresses a key business challenge: customers have different levels of engagement and commercial value, yet applying the same acquisition, retention and marketing strategy to all customers can result in inefficient allocation of resources.

The objective is therefore to develop a predictive analytics solution that can identify patterns associated with higher customer lifetime value and support more targeted customer segmentation, retention, marketing and revenue-generation strategies.

The project covers the complete machine-learning lifecycle, from exploratory data analysis and feature selection through model development, validation, explainability and deployment.

---

## 2. Business Problem

NexaSat serves customers with different demographic characteristics, transaction behaviours, payment preferences and levels of digital-wallet engagement.

A standardised marketing strategy does not necessarily reflect these differences. This can result in:

* Inefficient customer acquisition spending
* Poor targeting of retention campaigns
* Missed opportunities for upselling and cross-selling
* Limited understanding of the drivers of customer value
* Inefficient allocation of marketing resources

A predictive CLV model can provide a more data-driven approach by estimating customer value and identifying the behavioural characteristics associated with higher-value customers.

---

## 3. Project Aim

The aim of the project is to develop an **AI-driven Customer Lifetime Value prediction model** capable of estimating customer LTV from historical customer characteristics and behaviour.

The project combines:

* Exploratory data analysis
* Statistical feature analysis
* Feature selection
* Machine learning
* Cross-validation
* Model performance evaluation
* Explainable AI using SHAP
* Prediction/inference pipelines
* Streamlit application development
* AWS deployment
* Business intelligence reporting

---

## 4. Dataset

The dataset contains **7,000 customer observations**.

The data represents customer information from the NexaSat digital-wallet environment and includes demographic, behavioural, transactional and engagement-related information.

The prediction target is:

**LTV – Customer Lifetime Value**

The analysis focuses on identifying the characteristics and behavioural patterns that are associated with differences in customer lifetime value.

The dataset contained **no missing values** after preprocessing.

---

## 5. Exploratory Data Analysis

An initial exploratory analysis was conducted to understand the structure, composition and distribution of the customer dataset.

The analysis examined:

* Dataset dimensions
* Data types
* Missing values
* Duplicate observations
* Target-variable distribution
* Numerical-variable distributions
* Categorical-variable distributions
* Customer segmentation
* Relationships between customer characteristics and LTV

The analysis showed that customer LTV is **right-skewed**, with the mean LTV higher than the median.

The approximate:

* **Mean LTV:** 511,920
* **Median LTV:** 387,818

This indicates that a smaller number of high-value customers contribute substantially to the overall distribution of customer value.

---

## 6. Bivariate Analysis

Bivariate analysis was performed to investigate relationships between categorical customer characteristics and numerical variables.

Automated Plotly visualisations were used to compare numerical customer characteristics across categories.

The analysis helped identify whether customer value differed materially across groups such as:

* Income level
* Location
* Payment method
* App usage and engagement characteristics

The analysis indicated that demographic and categorical differences in average LTV were relatively modest compared with behavioural and transactional variables.

---

## 7. Feature Selection and Statistical Analysis

Several feature-analysis techniques were incorporated into the modelling workflow.

### Correlation Analysis

Correlation analysis was used to identify relationships between numerical predictors and to identify potentially redundant variables.

### Variance Inflation Factor

VIF analysis was used to assess potential multicollinearity among numerical predictors.

This helped reduce redundancy and improve the stability and interpretability of the modelling dataset.

### Mutual Information

Mutual information was also used to assess the strength of potentially nonlinear relationships between predictors and LTV.

Following feature analysis and removal of redundant engineered variables, the final modelling dataset contained **15 predictor variables**.

---

## 8. Data Preparation

The prepared dataset was separated into:

* **Features (X)**
* **Target variable (y = LTV)**

The data was divided into training and testing datasets using an **80/20 split**:

* Training observations: **5,600**
* Testing observations: **1,400**

A fixed random seed was used to ensure reproducibility.

The final model feature schema was saved so that the same feature structure could be applied during inference and deployment.

This is important for preventing **training-serving feature mismatch** when the model is used in production.

---

## 9. Machine Learning Models

Several machine-learning algorithms were evaluated to identify an appropriate model for predicting customer lifetime value.

The models included:

1. Linear Regression
2. Decision Tree
3. Random Forest
4. Support Vector Regression (SVR)
5. Multi-Layer Perceptron (MLP)
6. XGBoost
7. CatBoost
8. Tuned CatBoost

The models were evaluated using:

* R²
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Percentage Error (MAPE)

---

## 10. Model Validation

Model validation incorporated **5-fold cross-validation**.

The training data was divided into five folds, with each fold used sequentially as the validation set while the remaining folds were used for training.

Cross-validation was used to provide a more robust assessment of model performance and reduce reliance on a single training-validation split.

The final tuned CatBoost model was also evaluated on the independent test dataset.

---

## 11. Model Performance

The main model comparison was:

| Model              |        R² |        RMSE |         MAE |
| ------------------ | --------: | ----------: | ----------: |
| Linear Regression  |      0.44 |     329,543 |     243,560 |
| Decision Tree      |      0.74 |     226,900 |     151,245 |
| Random Forest      |      0.73 |     228,100 |     152,170 |
| XGBoost            |      0.71 |     235,673 |     155,766 |
| CatBoost           |      0.73 |     227,257 |     154,427 |
| MLP                |      0.42 |     335,000 |     250,653 |
| SVR                |     -0.04 |     450,229 |     340,844 |
| **Tuned CatBoost** | **0.737** | **226,407** | **152,767** |

The final tuned CatBoost model achieved:

* **R²: 0.737**
* **RMSE: 226,407**
* **MAE: 152,767**
* **MAPE: 55.8%**

The train-test R² gap was approximately **0.028**, indicating relatively limited deterioration between training and testing performance.

Approximately **74% of the variation in LTV** was explained by the final model.

---

## 12. Final Model – CatBoost

The final production model was a **tuned CatBoost Regressor**.

CatBoost was selected for the final implementation because it provided strong predictive performance within the model comparison exercise and captured nonlinear relationships between customer characteristics and lifetime value.

The model was tuned using **RandomizedSearchCV with 5-fold cross-validation**, resulting in 250 fitting operations during the search process.

The trained model was saved as:

```text
final_catboost_ltv_model.pkl
```

The corresponding model feature schema was saved as:

```text
models/model_features.pkl
```

---

## 13. Explainable AI – SHAP

Model explainability was incorporated using **SHAP (SHapley Additive exPlanations)**.

SHAP was used to understand both:

* Global feature importance
* The contribution of individual features to predictions

The SHAP analysis showed that:

### Spend per Active Day

**Spend per Active Day** was the strongest predictor of customer LTV.

Its influence was substantially greater than most of the other variables, with approximately **30 times the average influence of many other features**.

### Average Transaction Value

**Average Transaction Value** was the second strongest predictor.

This indicates that transactional intensity and customer spending behaviour are substantially more informative for predicting customer value than many demographic characteristics.

Other variables, including:

* Support tickets
* Loyalty points
* Payment method
* Customer satisfaction
* Cashback
* Age
* Income
* Referrals

had comparatively smaller individual contributions to the model.

---

## 14. Customer Segment Analysis

The analysis also examined differences in average LTV across customer segments.

### Income

Average LTV was approximately:

* Middle income: **522.9K**
* Low income: **510.1K**
* High income: **502.4K**

### Location

Average LTV was approximately:

* Suburban: **520.4K**
* Urban: **507.6K**
* Rural: **507.8K**

The differences across these demographic categories were relatively small compared with the influence of behavioural and transactional variables.

Payment-method and app-usage differences were also relatively modest, generally below approximately 3%.

---

## 15. Key Business Insight

The central finding from the modelling and explainability analysis is that **customer lifetime value is more strongly associated with transactional engagement than with basic demographic characteristics**.

In particular, spending intensity and transaction value were the strongest predictors identified by the model.

This suggests that NexaSat's customer-value strategy can benefit from focusing on **how customers use the wallet**, rather than relying heavily on demographic segmentation alone.

---

## 16. Business Applications

The model can support several customer-management activities.

### Customer Segmentation

Customers can be ranked according to predicted LTV and placed into value tiers.

### Retention

Customers showing declining transactional engagement can be identified for targeted retention interventions.

### Personalised Marketing

Marketing resources can be allocated according to predicted customer value and behavioural characteristics.

### Upselling and Cross-Selling

Higher-value customers can be identified for relevant product and service opportunities.

### Resource Allocation

Marketing and customer-management resources can be prioritised according to predicted customer value.

### Customer Monitoring

Changes in spending behaviour can be monitored as potential indicators of changes in future customer value.

---

## 17. Business Recommendations

Based on the modelling results, the project recommends:

1. **Increase transactional engagement** by encouraging customers to use the wallet more frequently.

2. **Increase transaction value** through relevant products, services and incentives.

3. **Monitor spend per active day** as an important customer-value indicator.

4. Use predicted LTV to **rank and tier customers** rather than treating all customers identically.

5. Avoid relying excessively on **income, location or other demographic characteristics** when designing customer-value strategies.

6. Periodically retrain and monitor the model as customer behaviour and market conditions change.

---

## 18. Prediction and Inference Pipeline

A separate `prediction.py` module was designed for production inference.

The inference pipeline is responsible for:

1. Receiving customer input.
2. Validating the input.
3. Loading the trained CatBoost model.
4. Loading the saved feature schema.
5. Applying the same feature engineering and encoding used during model development.
6. Aligning the input with the expected model features.
7. Converting values to the appropriate numerical format.
8. Generating the LTV prediction.
9. Returning the predicted customer lifetime value.

Separating prediction logic from the application prevents the Streamlit interface from containing duplicated modelling or preprocessing logic.

---

## 19. Streamlit Application

A Streamlit application was developed to provide an interactive interface for the trained model.

The application:

* Collects customer information
* Validates inputs
* Sends the information to the prediction pipeline
* Generates a predicted LTV
* Displays the result to the user
* Provides information about the model and analytical process

The application can be launched using:

```bash
streamlit run CLV_app.py
```

The application is designed to use the saved production model rather than retraining the model each time the application is launched.

---

## 20. AWS Deployment

The project is structured to support deployment to **AWS**.

The deployment architecture separates:

```text
Customer Input
      ↓
Streamlit Application
      ↓
Prediction Pipeline
      ↓
Saved CatBoost Model
      ↓
Predicted Customer LTV
```

The same trained model and feature schema used during development can therefore be used within the deployed application.

---

## 21. Power BI

The project can also be integrated with **Power BI** to provide business-facing customer analytics.

A dashboard can be used to present:

* Customer LTV distribution
* Predicted LTV
* Customer value segments
* Transactional behaviour
* Customer demographics
* Model performance
* Key customer-value drivers

This provides a bridge between machine-learning outputs and business decision-making.

---

## 22. Project Structure

```text
NexaSat _Digital _Wallet_Customer Life Time Value Prediction/
│
├── CLV_app.py
├── prediction.py
├── final_catboost_ltv_model.pkl
├── metrics.csv
├── requirements.txt
├── README.md
│
├── data/
│   └── preprocessed_CLV.csv
│
├── models/
│   └── model_features.pkl
│
├── notebooks/
│   └── NexaSat_CLV_Model.ipynb
│
├── outputs/
│   ├── figures/
│   └── tables/
│
└── tests/
    └── test_prediction.py
```

---

## 23. Technology Stack

### Programming and Data Analysis

* Python
* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* CatBoost
* XGBoost
* Scikit-learn
* TensorFlow / MLP

### Explainable AI

* SHAP

### Visualisation

* Plotly

### Application and Deployment

* Streamlit
* AWS

### Business Intelligence

* Power BI

### Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

### AI-Assisted Development

* ChatGPT
* Claude
* GitHub Copilot
* Microsoft Copilot
* Google Gemini

---

## 24. Limitations

Although the final model explains approximately 74% of the observed variation in LTV, approximately 26% remains unexplained.

The relatively high **55.8% MAPE** also indicates that individual customer-level LTV predictions should not be interpreted as precise revenue forecasts.

The model is therefore more suitable for:

* Customer ranking
* Value segmentation
* Relative prioritisation
* Identifying behavioural patterns

than for treating every individual prediction as an exact financial forecast.

Additional variables may potentially improve predictive performance, including:

* Customer tenure
* Acquisition channel
* Product mix
* Historical retention behaviour
* Macroeconomic indicators
* More detailed transaction history

The dataset also represents a point-in-time view, meaning model performance should be monitored as customer behaviour changes.

---

## 25. Future Development

Potential future improvements include:

* Additional behavioural and transactional features
* Customer tenure modelling
* Acquisition-channel analysis
* Product-level customer value modelling
* Time-series customer behaviour
* Customer churn integration
* LTV segmentation
* Automated model monitoring
* Model drift detection
* Automated retraining
* Cloud-based model serving
* Real-time prediction APIs
* Advanced AWS deployment
* Integration with Power BI
* Automated business reporting

---

## 26. End-to-End Project Workflow

The complete analytical workflow is:

```text
NexaSat Customer Dataset
          ↓
Data Validation
          ↓
Exploratory Data Analysis
          ↓
Univariate Analysis
          ↓
Bivariate Analysis
          ↓
Correlation Analysis
          ↓
VIF Analysis
          ↓
Mutual Information
          ↓
Feature Selection
          ↓
Train/Test Split
          ↓
Machine Learning Models
          ↓
5-Fold Cross-Validation
          ↓
Hyperparameter Tuning
          ↓
Model Evaluation
          ↓
Final CatBoost Model
          ↓
SHAP Explainability
          ↓
Business Interpretation
          ↓
Model Serialisation
          ↓
Prediction Pipeline
          ↓
Streamlit Application
          ↓
AWS Deployment
          ↓
Power BI Business Dashboard
```

---

## 27. Conclusion

The NexaSat Digital Wallet CLV project demonstrates an end-to-end machine-learning approach to customer lifetime value prediction.

Using a dataset of **7,000 customers**, multiple machine-learning algorithms were evaluated and compared. The final tuned CatBoost model achieved an **R² of 0.737**, with an **RMSE of 226,407**, **MAE of 152,767** and **MAPE of 55.8%** on the test dataset.

The explainability analysis showed that **Spend per Active Day** and **Average Transaction Value** were the most influential predictors of customer lifetime value. This provides an important business insight: transactional engagement appears to provide more predictive information about customer value than many demographic characteristics.

The project therefore demonstrates not only predictive modelling capability, but also the complete transition from **data analysis → feature engineering → machine learning → validation → explainability → business interpretation → deployment**.

The resulting solution provides a foundation for data-driven customer segmentation, retention, personalised marketing, upselling, cross-selling and customer-value management within a digital-wallet environment.
