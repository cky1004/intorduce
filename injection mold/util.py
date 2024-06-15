import collections
from math import sqrt
from enum import Enum

class ModelTask(Enum):
    BINARY_CLASSIFICATION = 0
    MULTICLASS_CLASSIFICATION = 1
    REGRESSION = 2


class ModelType(Enum):
    TENSORFLOW = 0
    SCIKIT_LEARN = 2
    PYTORCH = 3
    XGBOOST = 4


def get_model_metrics(
    model,
    task: ModelTask,
    phase: str,
    X,
    y_true,
):
    
    assert phase in ["train", "val", "test"]
    if task == ModelTask.REGRESSION:
        return _get_regressor_metrics(model, phase, X, y_true)

_SklearnMetric = collections.namedtuple(
    "_SklearnMetric", ["name", "function", "arguments"]
)

def _get_regressor_metrics(fitted_estimator, prefix, X, y_true):

    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
    )

    y_pred = fitted_estimator.predict(X)

    regressor_metrics = [
        _SklearnMetric(
            name=prefix + "_" + "mse",
            function=mean_squared_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
            ),
        ),
        _SklearnMetric(
            name=prefix + "_" + "mae",
            function=mean_absolute_error,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,
            ),
        ),
        _SklearnMetric(
            name=prefix + "_" + "r2_score",
            function=r2_score,
            arguments=dict(
                y_true=y_true,
                y_pred=y_pred,

            ),
        ),
    ]

    metrics_value_dict = _get_metrics_value_dict(regressor_metrics)
    metrics_value_dict[prefix + "_" + "rmse"] = sqrt(
        metrics_value_dict[prefix + "_" + "mse"]
    )

    return metrics_value_dict

def _get_metrics_value_dict(metrics_list):
    return {
        metric.name: float(metric.function(**metric.arguments))
        for metric in metrics_list
    }