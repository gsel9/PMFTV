"""
import pathlib
from contextlib import contextmanager

import mlflow
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from longimc import CMF, SCMF, WCMF
from longimc.factor_models.convergence import ConvergenceMonitor
from longimc.logging import (  # MLFlowLoggerArtifact,; MLFlowLoggerDiagnostic,;
MLFlowRunHierarchyException,; _aggregate_fields,; dummy_logger_context,
    MLFlowLogger,
)


def test_aggregate_fields():
    # Field values must be floats, so define some verbose variables with arbitrary
    # numerical values
    foo = 1.0
    bar = 2.0
    data = [
        {"field1": foo, "field2": foo},
        {"field1": foo, "field2": bar},
    ]
    correct_out = {
        "field1": foo,
        "field2_0": foo,
        "field2_1": bar,
        "field2_mean": np.mean((foo, bar)),
        "field2_std": np.std((foo, bar)),
    }
    out = _aggregate_fields(data)
    assert out == correct_out

    foo_string = "type1"
    bar_string = "type2"
    data = [
        {"field1": foo_string, "field2": foo_string},
        {"field1": foo_string, "field2": bar_string},
    ]
    correct_out = {
        "field1": foo_string,
        "field2_0": foo_string,
        "field2_1": bar_string,
        "field2_mean": float("nan"),
        "field2_std": float("nan"),
    }
    out = _aggregate_fields(data)
    assert set(correct_out) == set(out)
    for field, correct_value in correct_out.items():
        if isinstance(correct_value, float) and np.isnan(correct_value):
            assert np.isnan(out[field])
        else:
            assert out[field] == correct_value


def test_mlflow_context_hierarchy():

    with MLFlowLogger(allow_nesting=True):
        assert mlflow.active_run() is not None
    assert mlflow.active_run() is None

    with MLFlowLogger(allow_nesting=False):
        assert mlflow.active_run() is not None
    assert mlflow.active_run() is None

    with pytest.raises(MLFlowRunHierarchyException):
        with MLFlowLogger(allow_nesting=False):
            with MLFlowLogger(allow_nesting=False):
                pass
    assert mlflow.active_run() is None

    with pytest.raises(MLFlowRunHierarchyException):
        with MLFlowLogger(allow_nesting=False):
            assert mlflow.active_run() is not None
            with MLFlowLogger(allow_nesting=True):
                assert mlflow.active_run() is not None
                with MLFlowLogger(allow_nesting=False):
                    pass
    assert mlflow.active_run() is None

    with MLFlowLogger(allow_nesting=False):
        assert mlflow.active_run() is not None
        with MLFlowLogger(allow_nesting=True):
            assert mlflow.active_run() is not None
            with MLFlowLogger(allow_nesting=True):
                assert mlflow.active_run() is not None
    assert mlflow.active_run() is None


def test_mlflow_logger(tmp_path):
    mlflow.set_tracking_uri(tmp_path)
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir()
    experiment_id = mlflow.create_experiment("TestExperiment")
    mlflow.set_experiment(experiment_id)

    sample_size, time_span = 100, 40
    U = V = np.random.choice(np.arange(5), size=(sample_size, time_span))
    x_true = x_pred = np.random.choice(np.arange(5), size=(sample_size))
    p_pred = np.random.random(size=(sample_size, 4))
    dummy_output = {
        "params": {},
        "metrics": {},
        "tags": {},
        "meta": {
            "results": {
                "U": U,
                "V": V,
                "x_true": x_true,
                "x_pred": x_pred,
                "p_pred": p_pred,
            }
        },
    }

    # The dummy logger context should not activate a new mlflow run.
    with dummy_logger_context as logger:
        assert mlflow.active_run() is None
        logger(dummy_output)

    # MLFlowLogger should activate an outer run.
    with MLFlowLogger() as logger:
        outer_run = mlflow.active_run()
        assert outer_run is not None
        logger(dummy_output)
        with MLFlowLogger() as inner_logger:
            inner_run = mlflow.active_run()
            assert inner_run is not None
            assert inner_run.data.tags["mlflow.parentRunId"] == outer_run.info.run_id
            inner_logger(dummy_output)
    assert mlflow.active_run() is None

    with MLFlowLoggerArtifact(artifact_path=artifact_path) as logger:
        run_with_artifact = mlflow.active_run()
        logger(dummy_output)
    stored_artifact_path = _artifact_path_from_run(run_with_artifact)
    assert not any(stored_artifact_path.iterdir())  # The directory should be empty.

    with MLFlowLoggerDiagnostic(artifact_path=artifact_path) as logger:
        run_with_artifact = mlflow.active_run()
        logger(dummy_output)
    stored_artifact_path = _artifact_path_from_run(run_with_artifact)
    stored_artifacts = stored_artifact_path.glob("*")
    supposed_to_be_stored = {
        "basis_.pdf",
        "coefs_.pdf",
        "confusion_.pdf",
        "roc_auc_micro_.pdf",
        "certainty_plot.pdf",
    }
    assert supposed_to_be_stored == {file.name for file in stored_artifacts}


def test_train_and_log_params():
    # Some arbitrary data size
    sample_size, time_span = 100, 40
    X = np.random.choice(np.arange(5), size=(sample_size, time_span))

    hyperparams = {
        "shift_range": list(range(-2, 3)),
        "rank": 5,
        "lambda1": 1,
        "lambda2": 2,
        "lambda3": 3,
    }
    extra_metrics = {  # Some arbitrary extra metric to log
        "my_metric": lambda model: np.linalg.norm(model.X),
    }
    all_metrics = [*extra_metrics, "loss"]  # We set log_loss=True in train_and_log

    @contextmanager
    def logger_context():

        def logger(log_dict):
            for param in hyperparams:
                # Some params are numpy arrays, so use np.all
                assert np.all(hyperparams[param] == log_dict["params"][param])
            for metric in all_metrics:
                assert metric in log_dict["metrics"]

        yield logger

    train_and_log(
        X_test=X,
        X_train=X,
        extra_metrics=extra_metrics,
        logger_context=logger_context(),
        epoch_generator=ConvergenceMonitor(  # Set fewer epochs in order to be faster
            number_of_epochs=10,
            patience=2,
            epochs_per_val=2,
        ),
        log_loss=True,
        **hyperparams
    )


def test_value_error_loss_extra_metric():
    # Some arbitrary data size
    # sample_size, time_span = 100, 40
    # X = np.random.choice(np.arange(5), size=(sample_size, time_span))

    # with pytest.raises(
    #    ValueError,
    #    match=(
    #        "log_loss True and loss is in extra_metrics. "
    #        "This is illegal, as it causes name collision!"
    #    ),
    # ):
    #    train_and_log(
    #        X,
    #        X,
    #        extra_metrics={"loss": lambda x: None},
    #        log_loss=True,
    #    )
"""
