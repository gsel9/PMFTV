import numpy as np
import pandas as pd


def dgd_oracle(M):

	X_pred = np.round(M)

	return X_pred
	

def ffill(X_train):

	Xna = X_train.copy()
	Xna[X_train == 0] = np.nan
	Xfilled = pd.DataFrame(Xna).fillna(axis=1, method='ffill')
	X_pred = np.nan_to_num(Xfilled.values, 0)

	return X_pred
	