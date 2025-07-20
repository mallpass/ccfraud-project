# Credit Card Fraud Detection devops/mlops project
Matthew Allpass

## Goals
Hands on experience with techs like docker, supabase, render and CI/CD workflows
Simulate a ml system using versioned data and model retraining
Build end to end deployable project 

## Dataset
Found here: (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
~284,000 samples, very imbalanced dataset with only ~500 cases of fraud
Initial model trained on ~100,000 samples, the rest of the data was streamed in through the frontend to simuulate "real-time" retraining.

## Techs  
Frontend: REACT - handles new csv file uploads and displaying results
Backend: FastAPI - handles model and database use, input validation  
Database: PostgreSQL - hosted on supabase, handles sample storage for model (re)training  
ML Model: Keras - initial model trained with jupyter notebook, re-training proceedures baked into backend/mlops tech  
Devops: Github actions, Docker   
MLops: MLFlow  
Hosting: Render  
