import numpy as np
import pandas as pd

from simulation.utils.metrics import classification_report
from simulation.data.dataset import TrainTestData
from simulation.models.baseline import dgd_oracle, ffill


def main():

	prediction_window = 4

	M = np.load("/Users/sela/Desktop/tsd_code/data/dgd/40p/val/M_val.npy")
	X = np.load("/Users/sela/Desktop/tsd_code/data/dgd/40p/val/X_val.npy")
	
	test_data = TrainTestData(X=X, M=M, prediction_window=prediction_window)

	X_oracle_pred = dgd_oracle(test_data.M)
	oracle_scores = classification_report(test_data.X_test, test_data.O_test, X_oracle_pred, thresh=2)
	pd.Series(oracle_scores).to_csv("../../tmp_results/baseline/oracle.csv", header=False)

	X_ffill_pred = dgd_oracle(test_data.X_train)
	ffill_scores = classification_report(test_data.X_test, test_data.O_test, X_ffill_pred, thresh=2)
	pd.Series(ffill_scores).to_csv("../../tmp_results/baseline/ffill.csv", header=False)
	

if __name__ == "__main__":
	main()
