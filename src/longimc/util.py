import numpy as np
from longimc.logging import MLFlowLogger

# from sklearn.metrics import matthews_corrcoef


def fit_and_log(
    X: np.ndarray,
    W: np.ndarray,
    *,
    epoch_generator=None,
    dict_to_log=None,
    extra_metrics=None,
    log_loss: bool = True,
    logger_context=None,
    score_fn,
    **hyperparams,
):
    """Train model and log in MLFlow."""
    if logger_context is None:
        logger_context = MLFlowLogger()

    metrics = list(extra_metrics.keys()) if extra_metrics else []
    if log_loss:
        if "loss" in metrics:
            raise ValueError(
                "log_loss True and loss is in extra_metrics. "
                "This is illegal, as it causes name collision!"
            )
        metrics.append("loss")

    with logger_context as logger:
        # Create model
        # factoriser = model_factory(X_train, **hyperparams)

        # Fit model
        # results = factoriser.matrix_completion(
        #    extra_metrics=extra_metrics, epoch_generator=epoch_generator
        # )
        results = None

        mlflow_output: dict = {
            "params": {},
            "metrics": {},
            "tags": {},
            "meta": {},
        }

        # Score
        score = None  # score_fn(x_pred, x_true)
        results.update(
            {
                "score": score,
            }
        )
        mlflow_output["meta"]["results"] = results

        # Logging
        mlflow_output["params"].update(hyperparams)
        # mlflow_output["params"]["model_name"] =
        # #factoriser.config.get_short_model_name()
        if dict_to_log:
            mlflow_output["params"].update(dict_to_log)

        mlflow_output["metrics"]["matthew_score"] = score
        for metric in metrics:
            mlflow_output["metrics"][metric] = results[metric]
        logger(mlflow_output)
    return mlflow_output
