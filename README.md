# Credit Card Fraud Detection DevOps/MLOps Project
Matthew Allpass

## Goals
Hands on experience with techs like Docker, Supabase, Render, MLflow and CI/CD workflows
Simulate a machine learning retraining pipeline with "real" data coming in while the project is live
Build end to end deployable project 

## Dataset
Found here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Size: ~284,000 samples
Features: Time, Amount, V1-V28 (anonymized, PCA transformed values)
Class Distribution: Highly Imbalanced, of the ~284,000 samples <500 are fraudulent

The full dataset was split into 5 sections. 
The first 100,000 samples were used for initial model training.
The remaining samples were divided into 4 roughly equal portions, 3 of the 4 were fed in through the deployed app
and used for predictions and retraining. The remaining samples were held for model performance comparison. 

The initial model received an 85% AUPRC and a >99% accuracy, the high AUPRC score indicates the first model performed well
while the high accuracy ensures that the model was not making an egregious amount of false positives. 

Following training with the deployed system on the remaining 3 datasets, the dataset used here will be re-tested for comparison, 
and ideally improvement. 

## Tech Stack
Frontend: REACT - handles new csv file uploads and displaying results
Backend: FastAPI - handles model and database use, input validation  
Database: PostgreSQL - hosted on supabase, handles sample storage for model (re)training  
ML Model: Keras - initial model trained with jupyter notebook, re-training procedures baked into backend/mlops tech  
Devops: Github actions, Docker   
MLops: MLFlow  
Hosting: Render  

## Architecture and workflow
1. User uploads a CSV via React frontend
2. FastAPI validates and processes data. Data is stored in PostgreSQL database on supabase
3. Backend then strips uploaded data of labels, the current model is loaded and predictions are made
4. Prediction metrics are uploaded to the frontend

When data is uploaded to the database, a log is printed on render notifying of this new data. 
Retraining is then performed manually, locally, and simply by using the train.py script.
MLflow tracks the experiments and selects the best model, which I then push to GitHub for automated deployment. 

## Limitations, Problems, and Solutions

### Render Free Tier 
This is the biggest obstacle I ran into when implementing the original vision I had for the project. 
So far it has impacted 2 major components:

1. No automated retraining. Originally I had hoped to make it so that any update to the database would trigger retraining, with MLflow metrics being
printed to the deployment logs. However, unsuprisingly the roughly 500MB of memory given to Render free tier projects was not sufficient for training a
non linear logistic regression model, an XGBoost model, a random forest model, and a large neural network grid search.

The compromise I made was to just have the backend log a notification that the database had been updated. Then I'd locally run train.py and get the updated model and metrics into 
the repo. 

2. File size limits. Also unsurprisingly 500MB of memory was not enough to support a large amount of samples for predictions, as such I just limited the file size to 30MB.

### Machine Learning Models

I have never worked with out of the box models using scikit learn or XGBoost. Since I hoped to focus on learning and getting exposure to the infra side of things with this project I tried to strike a balance between 
what I know and what I don't when it comes to machine learning. 

The neural network class is "from scratch" and should hopefully reflect my understanding of how they work. I am familiar with the underlying principles of (and have implemented "from scratch" versions of) non linear logistic regression models and 
random forests before. I decided to spend the most time with the neural network implementation, and test the waters with the other model types using scikit learn. As such, none of the other models ever really come close to touching the neural network's performance
(certainly not the XGBoost), so swapping between model archetectures was not a problem I encountered while retraining. 

## TO DOS
The biggest missing piece is a deployed frontend with a finished UI. Eventually it will display prediction results following an upload, show model metrics in a dashboard, and have a demo mode with pre-selected files. Links and screenshots of it to come. 
Additionally, since retraining is done locally I will upload screenshots of the MLFlow dashboard as well. 
