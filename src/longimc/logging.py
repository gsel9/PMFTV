"""Logging utilities.

The logging is written in a general way, however at the moment only MLFlow is supported
as backend.
"""

import pathlib
import re
from contextlib import nullcontext
from typing import Callable, cast

import mlflow
import numpy as np

# An AggregationFunction takes a field name and list of values for that field, and
# returns a dictionary of fields aggregated from the values.
AggregationFunction = Callable[[str, list[float] | list[list[float]]], dict]


def mlflow_logger(log_data: dict) -> None:
    """Log results dictionary to MLFlow.

    Given a dictionary on the format below, add the run to mlflow.
    Assumes there to be an active MLFlow run!
    The run is typically started by
    [`MLFLowLogger`][matfact.model.logging.MLFlowLogger].

    Params and metrics should have values that are floats.
    Metric also accepts a list of floats, in which case they are interpreted as
    the metric value as a function of the epochs.
    ```python
    {
        "params": {"param1": value, "param2": value,},
        "metrics": {"metric1": value, "metric2": [...]},
        "tags": {},
        "meta": {},  # Data not logged to MLFLow
    }
    ```
    """
    for parameter, value in log_data["params"].items():
        mlflow.log_param(parameter, value)
    for metric, value in log_data["metrics"].items():
        if isinstance(value, list):
            for epoch, value_at_epoch in enumerate(value):
                mlflow.log_metric(metric, value_at_epoch, step=epoch)
        else:
            mlflow.log_metric(metric, value)
    mlflow.set_tags(log_data["tags"])


def _mean_and_std(
    field_name: str, values: list[str] | list[float] | list[list[float]]
) -> dict:
    """Return a dict with mean and standard deviation of the values.

    If the entries of values are lists, use the last element of each, i.e. the mean
    and std at the last epoch.
    """
    _values: list[float] | list[str]
    if isinstance(values[0], list):
        _values = [value[-1] for value in cast(list[list[float]], values)]
    else:
        _values = cast(list[float] | list[str], values)

    mean: float
    std: float
    if isinstance(_values[0], float):
        _values = cast(list[float], _values)
        mean = cast(float, np.mean(_values))
        std = cast(float, np.std(_values))
    else:
        mean = std = float("nan")
    return {
        f"{field_name}_mean": mean,
        f"{field_name}_std": std,
    }


def _store_subruns(field_name: str, values: list[float] | list[list[float]]) -> dict:
    return {f"{field_name}_{i}": value for i, value in enumerate(values)}


def _aggregate_fields(
    data: list[dict],
    aggregate_funcs: list[AggregationFunction] | None = None,
) -> dict:
    """Combine data for fields.

    If all entries have the same value for a given field, the output will have that
    field value combination. For fields with different values for the various
    runs, however, the output will log each value of that field.
    For example, if the field "field2" is foo in the first run and bar in the second,
    the output will have the fields "field1_0": foo, "field1_1": bar.

    Note that if the value is a list, it is interpreted as being a log over the epochs,
    and always taken to be unique for each run.

    `aggregate_func` is a function Callable[field_name: str, values: list] that adds
    extra fields for values that are not equal across runs. By default, mean and
    standard deviation.

    data = [
        {"field1": foo, "field2": foo, ...},
        {"field1": foo, "field2": bar, ...},
    ]

    -> {"field1": foo, "field2_0": foo, "field2_1": bar,}
    """
    if not data:  # The list is empty
        return {}

    if aggregate_funcs is None:
        aggregate_funcs = [_store_subruns, _mean_and_std]

    # All entries should have the same fields
    fields = set(data[0])
    for entry in data[1:]:
        assert set(entry) == fields

    new_data = {}
    for field in fields:
        values = [entry[field] for entry in data]
        # If all values are the same, no aggregation is needed
        should_aggregate = (
            isinstance(values[0], list | np.ndarray) or len(set(values)) > 1
        )
        if should_aggregate:  # Data different, we must aggregate
            for aggregation_function in aggregate_funcs:
                new_data.update(aggregation_function(field, values))
        else:  # All runs have the same data
            new_data[field] = values[0]

    return new_data


def batch_mlflow_logger(
    log_data: list[dict], aggregate_funcs: list[AggregationFunction] | None = None
) -> None:
    """Combine and log a set of runs.

    Used in for example cross validation training, where all folds should be logged
    as one run.

    Arguments:
        log_data: list of run data. Each entry in log_data should be compatible with the
            format expected by `_mlflow_logger`.
            ```python
            {
                "params": {"field1": value1,},
                "metrics": {"field2": foo, "field_history": [...],},
                "tags": {},
            }
            ```
    """
    new_log: dict = {
        "params": {},
        "metrics": {},
        "tags": {},
    }

    new_log["params"] = _aggregate_fields(
        [data["params"] for data in log_data], aggregate_funcs=aggregate_funcs
    )
    new_log["metrics"] = _aggregate_fields(
        [data["metrics"] for data in log_data], aggregate_funcs=aggregate_funcs
    )

    mlflow_logger(new_log)


class MLFlowRunHierarchyException(Exception):
    """MLFlowLogger contexts nested illegally."""

    pass


class MLFlowLogger:
    """Context manager for MLFlow logging.

    Wraps the code inside the corresponding with block in an MLFlow run.

    Arguments:
     allow_nesting: loggers can be nested within each other, with inside runs being
        logged as children in MLFlow.
     extra_tags: these tags will be appended to each run.

    Examples:

        ```python
        with MLFlowLogger() as logger:
            output = get_output_data()
            # output = {
            #     "params": {...},
            #     "metrics": {...},
            #     "meta": {...},
            # }
            logger(output)
        ```

    Raises:
        MLFlowRunHierarchyException: Raised on enter if
            loggers are nested when allow_nesting is False.

    See also:
        - [MLFlowBatchLogger][matfact.model.logging.MLFlowBatchLogger]
        - [MLFlowLoggerArtifact][matfact.model.logging.MLFlowLoggerArtifact]
        - [MLFlowLoggerDiagnostic][matfact.model.logging.MLFlowLoggerDiagnostic]
    """

    def __init__(self, allow_nesting: bool = True, extra_tags: dict | None = None):
        self.allow_nesting = allow_nesting
        self.extra_tags = extra_tags if extra_tags else {}

    def __enter__(self):
        try:
            self.run_ = mlflow.start_run(nested=self.allow_nesting)
        except Exception as e:
            if re.match("Run with UUID [0-9a-f]+ is already active.", str(e)):
                self.__exit__(type(e), str(e), e.__traceback__)
                raise MLFlowRunHierarchyException(
                    "allow_nesting is False, but loggers are nested!"
                )

        return self

    def __exit__(self, type, value, traceback):
        """End the run by calling the underlying ActiveRun's exit method."""
        if hasattr(self, "run_"):
            return self.run_.__exit__(type, value, traceback)
        else:
            return True

    def __call__(self, output_dict: dict):
        """Log an output dict to MLFlow."""
        mlflow_logger(output_dict)
        mlflow.set_tags(self.extra_tags)


class MLFlowBatchLogger(MLFlowLogger):
    """Context manager for combining multiple run data into one MLFlow run.

    Given several run dictionaries, the data is aggregated together to one
    summary run, which is logged to MLFlow. Used in for example cross validation runs,
    where each fold is a subrun, and the entire cross validation is logged as one run.

    Examples:
        It is possible to run MLFlowBatchLogger wrapped around subrun contexts.
        >>> with MLFlowBatchLogger() as outer_logger:
        >>>     for subrun in subruns:
        >>>         with MLFlowLogger() as inner_logger:
        >>>             ...
        >>>             inner_logger(run_data)
        >>>             outer_logger(run_data)  # (1)

        1. The data is only logged to MLFlow when `MLFlowBatchLogger` exists.

            ??? Note "How and when are things logged?"
                Each call to the logger is stored as a subrun, which
                on exit is aggregated together to one run which is logged to MLFLow.
                See [matfact.model.logging.batch_mlflow_logger][] for details.
    """

    def __init__(
        self,
        allow_nesting: bool = True,
        extra_tags: dict | None = None,
        aggregate_funcs: list[AggregationFunction] | None = None,
    ) -> None:
        super().__init__(allow_nesting=allow_nesting, extra_tags=extra_tags)
        self.aggregate_funcs = aggregate_funcs

    def __enter__(self):
        self.output = []
        return super().__enter__()

    def __exit__(self, type, value, traceback):
        batch_mlflow_logger(self.output, aggregate_funcs=self.aggregate_funcs)
        return super().__exit__(type, value, traceback)

    def __call__(self, output_dict):
        # On call we only append the dict to our list of data.
        # The actual logging happens on __exit__.
        self.output.append(output_dict)
        if self.extra_tags:
            mlflow.set_tags(self.extra_tags)


class MLFlowLoggerArtifact(MLFlowLogger):
    """Context manager for MLFlow logging, with artifact logging.

    All artifacts in artifact_path will be logged to the MLFlow run."""

    def __init__(
        self,
        artifact_path: pathlib.Path,
        allow_nesting: bool = True,
        extra_tags: dict | None = None,
        # create_artifact_path: bool = settings.paths.create_default,
    ):
        super().__init__(allow_nesting=allow_nesting, extra_tags=extra_tags)
        self.figure_path = artifact_path
        # if create_artifact_path:
        #    artifact_path.mkdir(parents=True, exist_ok=True)

    def __call__(self, output_dict):
        super().__call__(output_dict)
        mlflow.log_artifacts(self.figure_path)


dummy_logger_context = nullcontext(lambda _: None)  # Do-nothing logger
