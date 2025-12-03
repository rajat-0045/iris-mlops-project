import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("file:./mlruns")


def main():
    # Load data
    df = pd.read_csv("data/iris.csv")
    X = df.drop("Name", axis=1)
    y = df["Name"]

    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    mlflow.set_experiment("iris-mlops-project")

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.set_tag("note", "Baseline RandomForest on Iris")

        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)

        # Overall metrics
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="macro")
        recall = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", precision)
        mlflow.log_metric("recall_macro", recall)
        mlflow.log_metric("f1_macro", f1)

        # Per-class metrics
        report = classification_report(y_test, preds, output_dict=True)
        for class_name in ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]:
            mlflow.log_metric(f"{class_name}_precision", report[class_name]["precision"])
            mlflow.log_metric(f"{class_name}_recall", report[class_name]["recall"])
            mlflow.log_metric(f"{class_name}_f1", report[class_name]["f1-score"])

        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
