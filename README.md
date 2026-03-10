# Shipment Delay Prediction System

## 📊 Live Tableau Dashboard https://public.tableau.com/views/ShipmentDelayPredictionMLAnalyticsDashboard/ShipmentDelayAnalysis

## Overview
This project builds a machine learning system to predict shipment delays using historical logistics data. The goal is to help supply chain teams identify shipments that are likely to be delayed so they can take proactive action.
The project demonstrates a complete ML workflow including data pipelines, feature engineering, model training, evaluation, and API deployment.

## Problem
Shipment delays can disrupt logistics operations and affect delivery reliability. By analyzing historical shipment data, we can train a model that predicts delay risk and supports better operational planning.

## Project Highlights
- End-to-end machine learning pipeline
- Feature engineering for shipment data
- Baseline model using Logistic Regression
- Improved model using SMOTE + XGBoost
- Hyperparameter tuning with RandomizedSearchCV
- Real-time prediction API using FastAPI
- Interactive Tableau dashboard with 4 analytical views
- Data export pipeline generating dashboard-ready CSV datasets

## Model Results (Baseline)
Evaluation performed on 15,549 shipments.
Overall Accuracy: 58%
The model performs well at detecting delayed shipments, which is useful for identifying delivery risk.

## Model Improvement (XGBoost)
To improve the baseline Logistic Regression model, I implemented a more advanced approach using SMOTE and XGBoost.
The dataset had class imbalance, so SMOTE (Synthetic Minority Oversampling Technique) was used to balance the training data. After balancing the dataset, an XGBoost classifier was trained to capture more complex relationships in the shipment features.
Hyperparameter tuning was performed using RandomizedSearchCV to find the best model parameters. This improved model provides better learning capability compared to the baseline model and demonstrates a more production-oriented machine learning workflow.

## API Prediction 

##### Request:
{
  "payment_type": "DEBIT",
  "profit_per_order": 34.448338,
  "sales_per_customer": 92.49099,
  "category_id": 9.0,
  "category_name": "Cardio Equipment",
  "customer_city": "Caguas",
  "customer_country": "Puerto Rico",
  "customer_id": 12097.683,
  "customer_segment": "Consumer",
  "customer_state": "PR",
  "customer_zipcode": 725.0,
  "department_id": 3.0,
  "department_name": "Footwear",
  "latitude": 18.359064,
  "longitude": -66.370575,
  "market": "Europe",
  "order_city": "Viena",
  "order_country": "Austria",
  "order_customer_id": 12073.336,
  "order_date": "2015-08-12 00:00:00+01:00",
  "order_id": 15081.289,
  "order_item_cardprod_id": 191.0,
  "order_item_discount": 12.623338,
  "order_item_discount_rate": 0.13,
  "order_item_id": 38030.996,
  "order_item_product_price": 99.99,
  "order_item_profit_ratio": 0.41,
  "order_item_quantity": 1.0,
  "sales": 99.99,
  "order_item_total_amount": 84.99157,
  "order_profit_per_order": 32.083145,
  "order_region": "Western Europe",
  "order_state": "Vienna",
  "order_status": "COMPLETE",
  "product_card_id": 191.0,
  "product_category_id": 9.0,
  "product_name": "Nike Men's Free 5.0+ Running Shoe",
  "product_price": 99.99,
  "shipping_date": "2015-08-13 00:00:00+01:00",
  "shipping_mode": "Standard Class",
  "ship_year": 2015,
  "ship_month": 8,
  "ship_dayofweek": 3
}

##### Response:
{
  "prediction": 1,
  "probability_delay": 0.1971648178573689
}
## Tableau Dashboard

Interactive 4-dashboard analytics suite:

| Dashboard | What It Shows |
|-----------|--------------|
| Executive Overview | Shipment KPIs, delay rate by market, monthly trend |
| Model Performance | Baseline vs XGBoost side-by-side comparison |
| Prediction Insights | XGBoost feature importance, prediction distribution |
| Business Impact | Delay patterns by segment, shipping mode, market |

## Tech Stack
- Python
- Pandas
- Scikit-Learn
- XGBoost
- FastAPI
- SMOTE
- Git / GitHub
- Tableau Public (dashboard)


