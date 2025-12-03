# iris-mlops-project

End-to-end MLOps mini project using the Iris dataset.  
It includes model training with MLflow tracking, automated training on GitHub Actions, and a FastAPI API for predictions.

## Project structure

- `data/iris.csv` – Iris dataset used for training.
- `src/train.py` – Trains a RandomForest model, logs accuracy with MLflow.
- `src/predict.py` – FastAPI app that serves a `/predict` endpoint.
- `.github/workflows/train.yml` – GitHub Actions workflow to run training on each push.

## Setup and local training

1. Create and activate virtualenv (Windows):

