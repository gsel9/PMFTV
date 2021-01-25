import numpy as np

from scipy.stats import betabinom
from sklearn.model_selection import train_test_split

from mask import simulate_mask
from dgd_generator import simulate_float_from_named_basis, simulate_dgd


SEED = 42


def censoring(Y, missing=0):

	t_cens = betabinom.rvs(n=Y.shape[1], a=4.57, b=5.74, size=Y.shape[0])

	for i, t_end in enumerate(t_cens):
		Y[i, t_end:] = missing

	return Y


def produce_dataset(N, T, r, seed, level, memory_length=10, missing=0):

	M = simulate_float_from_named_basis(
		basis_name='simple_peaks', 
		N=N, 
		T=T, 
		K=r, 
		domain=[1, 4], 
		random_state=seed
	)

	D = simulate_dgd(
		M, 
		domain_z=np.arange(1, 5),
		theta=2.5,
		random_state=seed
	)

	O = simulate_mask(
		D,
		screening_proba=np.array([0.05, 0.15, 0.4, 0.6, 0.2]),
		memory_length=memory_length,
		level=level,
		random_state=seed
	)

	Y = D * O
	Y = censoring(Y, missing=missing)

	valid_rows = np.count_nonzero(Y, axis=1) > 1

	return M[valid_rows], Y[valid_rows]


def main():

	M, Y = produce_dataset(500, T=50, r=5, seed=SEED, level=3, memory_length=10)

	np.save("../../../../data/synthetic/M.npy", M)
	np.save("../../../../data/synthetic/Y.npy", Y)


if __name__ == "__main__":
	main() 
