# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import logging
import os
import sys
import warnings
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from ydata_profiling import ProfileReport
from trail import Trail

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = ("winequality-red.csv")
    try:
        data = pd.read_csv(csv_url, sep=";")
        print(data.head())
        print("data")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    PROFILE_PATH = "train_data_report.html"
    profile = ProfileReport(train_x, title="Ydata Profiling Report")
    profil = profile.to_file(PROFILE_PATH)

    with mlflow.start_run():
        with Trail("myProjectAlias") as trail:
            trail.put_hypothesis("next step")
            trail.put_artifact(PROFILE_PATH, "Train Data Profile", "data")

            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            predicted_qualities = lr.predict(test_x)

            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

            print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(
                alpha, l1_ratio))
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
