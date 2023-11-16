"""
import pathlib
from itertools import product

import tensorflow as tf
from matfact.data_generation import Dataset
from matfact.model import reconstruction_mse, train_and_log
from matfact.model.logging import MLFlowLoggerDiagnostic
from matfact.settings import settings


def get_objective(**hyperparams):

    X_train, X_test, *_ = data.get_split_X_M()

    @use_named_args(search_space)
    def objective(**search_hyperparams):
        hyperparams.update(search_hyperparams)
        mlflow_output = train_and_log(
            X_train,
            X_test,
            logger_context=MLFlowLogger(),
            dict_to_log=data.prefixed_metadata(),
            log_loss=False,
            **hyperparams,
        )
        # The score logged, the Matthew correlation coefficient, is 'higher is
        # better', while we are minimizing.
        return -mlflow_output["metrics"]["matthew_score"]

    return objective


def example_hyperparameter_search():
    tf.config.set_visible_devices([], "GPU")
    mlflow.set_tracking_uri(settings.paths.base / "mlruns")
    space = (
        Real(-5.0, 1, name="lambda1"),
        Real(8, 20, name="lambda2"),
        Real(0.0, 20, name="lambda3"),
    )

    # Load data
    try:
        data = Dataset.from_file(settings.paths.dataset)
    except FileNotFoundError:  # No data loaded
        data = Dataset.generate(1000, 40, 5, 5)

    with mlflow.start_run():
        res_gp = gp_minimize(
            objective_getter(
                data, search_space=space, use_convolution=False, shift_range=None
            ),
            space,
            n_calls=10,
        )

        best_values = res_gp["x"]
        best_score = res_gp["fun"]
        # We are minimizing, so the best_score is inverted.
        mlflow.log_metric("best_score", -best_score)
        for param, value in zip(space, best_values):
            mlflow.log_param(f"best_{param.name}", value)
        mlflow.set_tag("Notes", "Hyperparameter search")


def get_objective_CV(
    **hyperparams,
):
    kf = KFold(n_splits=n_splits)
    X, _ = data.get_X_M()
    logger_context = MLFlowLogger() if log_folds else dummy_logger_context

    @use_named_args(search_space)
    def objective(**search_hyperparams):
        hyperparams.update(search_hyperparams)
        scores = []
        with MLFlowBatchLogger() as logger:
            for train_idx, test_idx in kf.split(X):
                mlflow_output = train_and_log(
                    X[train_idx],
                    X[test_idx],
                    dict_to_log=data.prefixed_metadata(),
                    logger_context=logger_context,
                    log_loss=False,
                    **hyperparams,
                )
                # The score logged, the Matthew correlation coefficient, is 'higher is
                # better', while we are minimizing.
                logger(mlflow_output)
                scores.append(-mlflow_output["metrics"]["matthew_score"])
        return np.mean(scores)

    return objective


def experiment(
    hyperparams,
    enable_shift: bool = False,
    enable_weighting: bool = False,
    enable_convolution: bool = False,
    mlflow_tags: dict | None = None,
    dataset_path: pathlib.Path = settings.paths.dataset,
):
    # Setup and loading #
    dataset = Dataset.from_file(dataset_path)
    X_train, X_test, M_train, _ = dataset.get_split_X_M()

    shift_range = list(range(-12, 13)) if enable_shift else []

    extra_metrics = {
        "recMSE": lambda model: reconstruction_mse(M_train, model.X, model.M),
    }

    train_and_log(
        X_train,
        X_test,
        shift_range=shift_range,
        use_weights=enable_weighting,
        extra_metrics=extra_metrics,
        use_convolution=enable_convolution,
        logger_context=MLFlowLoggerDiagnostic(
            settings.paths.figure, extra_tags=mlflow_tags
        ),
        **hyperparams,
    )


def main():
    # Generate some data
    Dataset.generate(N=1000, T=50, rank=5, sparsity_level=6).save(
        settings.paths.dataset
    )

    USE_GPU = False
    if not USE_GPU:
        tf.config.set_visible_devices([], "GPU")

    mlflow_tags = {
        "Developer": "Thorvald M. Ballestad",
        "GPU": USE_GPU,
        "Notes": "tf.function commented out",
    }
    # NB! lamabda1, lambda2, lambda3 does *not* correspond directly to
    # the notation used in the master thesis.
    hyperparams = {
        "rank": 5,
        "lambda1": -0.021857774198331015,
        "lambda2": 8,
        "lambda3": 4.535681885641427,
    }

    for shift, weight, convolve in product([False, True], repeat=3):
        experiment(
            hyperparams,
            shift,
            weight,
            convolve,
            mlflow_tags=mlflow_tags,
        )


if __name__ == "__main__":
    main()
"""
