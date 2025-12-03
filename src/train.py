import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")

def main():
    # Load data
    df = pd.read_csv("data/iris.csv")
    X = df.drop("Name", axis=1)
    y = df["Name"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_experiment("iris-mlops-project")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.set_tag("note", "Baseline RandomForest on Iris")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)
        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
