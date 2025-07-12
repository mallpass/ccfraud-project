# ccfraud-project


## Project Overview
devops/mlops project by Matthew Allpass

goal is to get more hands on experience with ci/cd, docker, deployment, and to delve into mlops 
dataset used is found here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
neural network model v1 will be made with ~100k samples of the ~300k total samples found in the dataset, the rest will be fed in through the front end to simulate new, "real" data

Frontend: REACT - handles new csv file uploads
Backend: FastAPI - handles model use, light data processing, and storing data in db
Database: PostgreSQL - hosted on supabase, handles sample storage for model retraining
ML Model: Keras - initial model trained with jupyter notebook, re-training proceedures baked into backend/mlops tech
Devops: Github, Docker 
MLops: MLFlow
Hosting: Render
