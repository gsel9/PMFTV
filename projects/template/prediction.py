import os 

import numpy as np

from simulation.data.dataset import TrainTestDataset
from simulation.models.inference.map import MAP

def predict_in_dir():

	path_to_results = "/Users/sela/Desktop/tsd_code/results/mf_tv/init1"

	# Update for each file name.
	split_idx = 3

	X_test = np.load("/Users/sela/Desktop/tsd_code/data/screening_filtered/test/X_test.npy")
	test_data = TrainTestDataset(X_test, time_lag=4)

	for fname in os.listdir(path_to_results):

		if "M_hat" in fname:
			M_hat = np.load(f"{path_to_results}/{fname}")

			estimator = MAP(M_train=M_hat, theta=2.5)
			y_pred = estimator.predict(test_data.X_train, test_data.time_of_prediction)
			y_pred_proba = estimator.predict_proba(test_data.X_train, test_data.time_of_prediction)

			label = ("_").join(fname.split("_")[:split_idx])
			np.save(f"{path_to_results}/{label}_y_true.npy", test_data.y_true)
			np.save(f"{path_to_results}/{label}_y_pred.npy", y_pred)
			np.save(f"{path_to_results}/{label}_y_pred_proba.npy", y_pred_proba)


def ensemble_prediction():
	# Aggregate model predictions in ensemble.

	X_test = np.load("/Users/sela/Desktop/tsd_code/data/screening_filtered/test/X_test.npy")
	test_data = TrainTestDataset(X_test, time_lag=4, random_state=42)

	path_to_predicted_probas = [
		"/Users/sela/Desktop/tsd_code/results/mf_rank_no_early_stopping"
		# recsys + lasso
	]

	predicted_probas = ...
	for path_to_result in path_to_results:
		pass

		# Aggregate predicted proba for each model.		

		#y_pred = estimator.predict(test_data.X_train, test_data.time_of_prediction)

	# * take argmax of proba from each model to get final predictions
	#label = ("_").join(fname.split("_")[:3])
	#np.save(f"{path_to_results}/{label}_y_true.npy", test_data.y_true)
	#np.save(f"{path_to_results}/{label}_y_pred.npy", y_pred)


if __name__ == '__main__':
	predict_in_dir()
	#ensemble_prediction()
