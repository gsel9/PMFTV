import numpy as np

import hashlib


class MAP:

    def __init__(self, M_train, theta=2.5, domain_z=np.arange(1, 5), binary_thresh=2):

        self.M_train = M_train
        self.theta = theta
        self.domain_z = domain_z

        self.z_to_binary_mapping = lambda z: np.array(z) > binary_thresh 

        # Initialize prediction probabilities
        self.__proba_z_precomputed = None
        self.__ds_X_hash = None
        self.__ds_t_hash = None

    def __is_match_ds_hash(self, X, t):
        """Check if hash of (X, t) matches stored

        Checks if the stored hexadecimal hash
        matches the hexademical hash of the input 
        (X, t). 

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        Returns
        -------
        match : bool
        True if match
        """
        if self.__ds_X_hash is None or self.__ds_t_hash is None:
            return False

        if (hashlib.sha1(X).hexdigest() == self.__ds_X_hash) and (hashlib.sha1(t).hexdigest() == self.__ds_t_hash):
            return True

        return False

    def __store_ds_hash(self, X, t):
        """Store hash of dataset.

        Stores a hexadecimal hash of the dataset X used
        in predict_proba.

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        Returns
        -------
        self
        Model with stored hash
        """
        self.__ds_X_hash = hashlib.sha1(X).hexdigest()
        self.__ds_t_hash = hashlib.sha1(t).hexdigest()

        return self

    def _loglikelihood(self, X):
        """Compute loglikelihood of X having originated from 
        the fitted profiles (U V^T).

        For all x_i in X, compute the log of the estimated
        likelihood that x_i originated from m_j for j = 1, ..., N
        where N = n_samples_train is the number of samples used in
        training the model. 

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        Returns
        -------
        logL : array_like, shape (n_samples, n_samples_train)
        The logs of the estimated likelihoods.
        """

        N_1 = self.M_train.shape[0]
        N_2 = X.shape[0]

        logL = np.ones((N_2, N_1))

        for i in range(N_2):
            row_nonzero_cols = X[i] != 0
            eta_i = (X[i, row_nonzero_cols])[None, :] - self.M_train[:, row_nonzero_cols]
            logL[i] = np.sum(-self.theta*np.power(eta_i, 2), axis=1)

        return logL

    def predict_proba(self, X, t):
        """Compute class probabilities.

        For all (x_i, t_i) in (X, t), compute the estimated
        probability that row i will at time t be in the state
        z for z in the domain_z of the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        Returns
        -------
        proba_z_normalized
        The probalities
        """

        # If evaluating several scoring methods subsequently,
        # significant computational time can be saved by storing
        # the class probabilities

        if self.__is_match_ds_hash(X, t):
            return self.__proba_z_precomputed

        X, t = X, t

        logL = self._loglikelihood(X)

        proba_z = np.empty((X.shape[0], self.domain_z.shape[0]))
        for i in range(X.shape[0]):
            proba_z[i] = np.exp(logL[i]) @ np.exp(-self.theta * (self.M_train[:, t[i], None] - self.domain_z)**2)

        # Normalize.
        proba_z_normalized = proba_z / (np.sum(proba_z, axis=1))[:, None]

        # Store probabilities
        self.__proba_z_precomputed = proba_z_normalized
        self.__store_ds_hash(X, t)

        return proba_z_normalized

    def predict_proba_binary(self, X, t):
        """Compute binary probabilities.

        For all (x_i, t_i) in (X, t), compute the estimated
        probability that row i will at time t be True.

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        Returns
        -------
        proba_bin
        The probalities.
        """
        # If evaluating several scoring methods subsequently,
        #  significant computational time can be saved by storing
        #  the class probabilities
        proba_z = self.predict_proba(X, t)

        values_of_z_where_true = [self.z_to_binary_mapping(z) for z in self.domain_z]

        proba_bin = np.sum(proba_z[:, values_of_z_where_true], axis=1).flatten()

        return proba_bin

    def predict(self, X, t, bias_z=None):
        """Predict the most probable state z at time t_i for each (x_i, t_i) in (X, t).

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        bias : array_like, shape (n_states_z, )
        The bias of the model.

        Returns
        -------
        z_states : (n_samples, )
        The predicted states.
        """
        proba_z = self.predict_proba(X, t)

        if bias_z is None:
            return self.domain_z[np.argmax(proba_z, axis=1)]
        else:
            return self.domain_z[np.argmax(proba_z * bias_z, axis=1)]

    def predict_binary(self, X, t, bias_bin=None):
        """Predict future binary outcome.

        For all (x_i, t_i) in (X, t), predict the most probable
        binary outcome at time t.

        Parameters
        ----------
        X : array_like, shape (n_samples, time_granularity)
        The regressor set.

        t : array_like, shape (n_samples, )
        Time of prediction.

        bias : array_like, shape (n_states_e, )
        The bias of the model.

        Returns
        -------
        bin_states : (n_samples, )
        The predicted states.
        """

        proba_bin = self.predict_proba_binary(X, t)

        if bias_bin is None:
            return np.ones_like(proba_bin) * (proba_bin >= 0.5)
        else:
            return np.ones_like(proba_bin) * (proba_bin >= 1 - bias_bin)

    def predict_risk(self, M):

        # Single test sample.
        if np.ndim(M) < 2:

            i = np.argmin(np.linalg.norm(self.M_train - M, axis=1) ** 2)

            return self.M_train[i]

        idx = [
            np.argmin(np.linalg.norm(self.M_train - m, axis=1) ** 2) for m in M
        ]

        return self.M_train[idx]


def ffill_clf(test_data):

    Xna = test_data.copy()
    Xna[test_data == 0] = np.nan
    Xfilled = pd.DataFrame(Xna).fillna(axis=1, method='ffill')
    X_pred = np.nan_to_num(Xfilled.values, 0)

    return X_pred


if __name__ == "__main__":

    import pandas as pd
    from tqdm import tqdm
    from simulation.dataset import TemporalDatasetPredict

    for model in ["ffill"]: #["mf_no_conv", "gdl", "mf"]:
        for data_type in ["screening_filtered"]:
            for forecast, time_lag in [("5yf", 20), ("7yf", 28)]:#[("1yf", 4), ("2yf", 8), ("3yf", 12)]:

                #X_test = np.load(f"/Users/sela/Desktop/recsys_paper/data/{data_type}/{density}/test/X_test.npy")
                #M = np.load(f"/Users/sela/Desktop/recsys_paper/data/{data_type}/{density}/test/M_test.npy")
                #test_data = TemporalDatasetPredict(X_test, ground_truth=M, time_lag=time_lag, random_state=42)

                X_test = np.load(f"/Users/sela/Desktop/recsys_paper/data/{data_type}/test/X_test.npy")
                test_data = TemporalDatasetPredict(X_test, time_lag=time_lag, random_state=42)

                #X_rec = np.load(f"/Users/sela/Desktop/recsys_paper/results/{data_type}/{model}/{density}/train/train_Xrec.npy")
                #X_rec = np.load(f"/Users/sela/Desktop/recsys_paper/results/{data_type}/{model}/train/train_Xrec.npy")
                #estimator = Predict(M_train=X_rec, theta=2.5)
                
                #y_pred = estimator.predict(test_data.X_train, test_data.time_of_prediction)
                #y_pred_proba = estimator.predict_proba(test_data.X_train, test_data.time_of_prediction)
                #y_pred_proba_binary = estimator.predict_proba_binary(test_data.X_train, test_data.time_of_prediction)

                # FF
                Y_pred = ffill_clf(test_data.X_train)
                # Oracle
                #Y_pred = np.round(test_data.M)
                y_pred = Y_pred[range(test_data.X_train.shape[0]), test_data.time_of_prediction]

                #path_to_results = f"/Users/sela/Desktop/recsys_paper/results/{data_type}/{model}/{density}/test"
                path_to_results = f"/Users/sela/Desktop/recsys_paper/results/{data_type}/{model}/test/{forecast}"
      
                np.save(f"{path_to_results}/y_true.npy", test_data.y_true)
                np.save(f"{path_to_results}/y_pred.npy", y_pred)
                #np.save(f"{path_to_results}/y_pred_proba.npy", y_pred_proba)
                #np.save(f"{path_to_results}/y_pred_proba_binary.npy", y_pred_proba_binary)
                