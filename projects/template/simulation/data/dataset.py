from abc import ABC

import numpy as np
import sklearn.model_selection


class DatasetBase(ABC):

    def __init__(self, data):
        self.X = np.array(data)


class TrainDataset(DatasetBase):

    def __init__(self, data, ground_truth=None):
    
        DatasetBase.__init__(self, data)
        self.ground_truth = np.array(ground_truth)

    @property
    def X_train(self):
        return self.X

    @property
    def ground_truth_train(self):
        return self.ground_truth


class TrainTestDataset(DatasetBase):

    def __init__(self, data, ground_truth=None, time_lag=4):

        DatasetBase.__init__(self, data)

        self.X = data
        self.time_lag = time_lag

        # Find time of last observed entry for all rows
        time_of_prediction = self.X.shape[1] - np.argmax(self.X[:, ::-1] != 0, axis=1) - 1

        # Copy values to be predicted
        y_true = np.copy(self.X[range(self.X.shape[0]), time_of_prediction])

        # Remove observations in or after prediction window
        for i_row in range(self.X.shape[0]):
            self.X[i_row, max(0, time_of_prediction[i_row] - time_lag):] = 0

        # Find rows that still contain observations
        self.valid_rows = np.sum(self.X, axis=1) > 0

        # Remove all rows that don't satisfy the specified criteria
        self.y = y_true[self.valid_rows]
        self.X = self.X[self.valid_rows]

        self.time_of_prediction = time_of_prediction[self.valid_rows]

        if ground_truth is not None:
            self.M  = ground_truth[self.valid_rows]

    @property
    def X_train(self):
        return self.X

    @property
    def y_true(self):
        return self.y

    @property
    def ground_truth_pred(self):
        return self.ground_truth


class KFoldDataset(DatasetBase):

    def __init__(self, data, ground_truth=None, time_lag=4, n_splits=5):

        self.ground_truth = ground_truth

        self.time_lag = time_lag
        self.n_splits = n_splits

        self.test_data = TrainTestDataset(data=data, time_lag=self.time_lag)
        self.train_data = TrainDataset(data=data[self.test_data.valid_rows])

        kfolds = sklearn.model_selection.KFold(n_splits, shuffle=False)
        self.idx_per_fold = [idx for idx in kfolds.split(self.train_data.X)]

        # Instantiate with 1st fold
        self.__i_fold = 0
        self.__idc_train, self.__idc_pred = self.idx_per_fold[self.i_fold]

    @property
    def train_rows_idx(self):
        return self.__idc_train

    @property
    def X_train(self):
        return self.train_data.X[self.__idc_train]

    @property
    def X_test(self):
        return self.test_data.X[self.__idc_pred]

    @property
    def y_true(self):
        return self.test_data.y_true[self.__idc_pred]

    @property
    def time_of_prediction(self):
        return self.test_data.time_of_prediction[self.__idc_pred]

    @property
    def i_fold(self):
        return self.__i_fold

    @i_fold.setter
    def i_fold(self, i_fold):

        assert int(i_fold) < self.n_splits

        self.__i_fold = int(i_fold)
        self.__idc_train, self.__idc_pred = self.idx_per_fold[self.__i_fold]
