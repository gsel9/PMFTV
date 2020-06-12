# TODO: Rename to along lines of mf_lars_rank_exp

import os 
import json
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 

#from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import matthews_corrcoef, pairwise_distances
from scipy.stats import wasserstein_distance
#from sklearn.cluster import KMeans


def loglikelihood(X, M_train, theta=2.5):

    N_1 = M_train.shape[0]
    N_2 = X.shape[0]

    logL = np.ones((N_2, N_1))
    for i in range(N_2):
        row_nonzero_cols = X[i] != 0
        eta_i = (X[i, row_nonzero_cols])[None, :] - M_train[:, row_nonzero_cols]
        logL[i] = np.sum(-1.0 * theta * np.power(eta_i, 2), axis=1)

    return logL


def train_test(X, time_lag=4, prediction_rule="last_observed", random_state=42):
    
    time_of_prediction = X.shape[1] - np.argmax(X[:, ::-1] != 0, axis=1) - 1

    y = np.copy(X[range(X.shape[0]), time_of_prediction])
    
    X_train = X.copy()
    for i_row in range(X.shape[0]):
        X_train[i_row, max(0, time_of_prediction[i_row] - time_lag):] = 0
        
    X_test = np.zeros_like(X_train)
    X_test[range(X_train.shape[0]), time_of_prediction] = y

    valid_rows = np.sum(X_train, axis=1) > 0

    return X_train[valid_rows], X_test[valid_rows], time_of_prediction[valid_rows]


def load_results(path_to_file):

    _, ext = os.path.splitext(path_to_file)

    if ext == ".pkl":

        with open(path_to_file, "rb") as infile:
            return pickle.load(infile)

    if ext == ".json":

        with open(path_to_file, "r") as infile:
            return json.load(infile)

    raise ValueError(f"Unknown file format {ext}")


def factorise(Z, rank, i=None):
    
    W, s, H = np.linalg.svd(Z, full_matrices=False)

    S = np.diag(np.sqrt(s[:rank]))

    U = np.dot(W[:, :rank], S)
    V = np.dot(S, H[:rank, :]).T
    
    if not np.allclose(Z, U @ V.T):
        print("---!!!--- Not close ---!!!---")
        if i is not None:
            print(i)
    
    return U, V


def get_results(path_to_results):

	configs = {}

	Us = {}
	Vs = {}
	M_hats = {}

	y_preds = {}
	y_trues = {}

	key = "run_param"
	split_idx = 2

	for fname in os.listdir(path_to_results):

		if key in fname:

			combo = fname.split("_")[split_idx]

			if "configs" in fname: 
				configs[combo] = load_results(f"{path_to_results}/{fname}")

			if "U" in fname:
				Us[combo] = np.load(f"{path_to_results}/{fname}")

			if "V" in fname:
				Vs[combo] = np.load(f"{path_to_results}/{fname}")

			if "M_hat" in fname:
				M_hats[combo] = np.load(f"{path_to_results}/{fname}")

			# NB: Be specific to avoid mixing y_pred and y_pred_proba.
			if "y_pred.npy" in fname:
				y_preds[combo] = np.load(f"{path_to_results}/{fname}").astype(float)

			if "y_true" in fname:
				y_trues[combo] = np.load(f"{path_to_results}/{fname}").astype(float)

	mcc = {combo: matthews_corrcoef(y_true, y_preds[combo]) for combo, y_true in y_trues.items()}

	return configs, M_hats, y_preds, y_trues, mcc, Us, Vs


def zero_threshold_coefs(Z, thresh=1e-7):

	_Z = Z.copy()
	_Z[((-thresh < Z) & (Z < thresh))] = 0

	return _Z


def active(Z, thresh=1e-7):
    
    zero = ((-thresh < Z) & (Z < thresh)).astype(int)
    
    nonzero = np.ones_like(zero)
    nonzero[zero == 1] = 0
    
    return nonzero


def plot_reconstructed(combos, M_hats, X_train, path_to_file):

	for combo in combos:

		X_rec = M_hats[combo]

		_, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 8))
		for num, axis in enumerate(axes.ravel()):

		    x_train = X_train[num].copy()
		    test_win = np.argmax(np.cumsum(x_train))
		    x_train[x_train == 0] = np.nan
		    
		    axis.plot(x_train, "o", label="train")
		    axis.plot(X_rec[num, :test_win])
		    axis.legend()

		plt.tight_layout()

		plt.savefig(path_to_file + "_" + str(combo) + ".pdf")


def plot_predictions(combos, M_hats, X_test, y_preds, y_trues, path_to_file):

	X_train, X_test, _ = train_test(X_test, 4)

	for combo in combos:

		X_rec = M_hats[combo]
		_y_pred = y_preds[combo]
		_y_true = y_trues[combo]
	
		match_idx = np.argmax(loglikelihood(X_train, X_rec), axis=1)
		_X_rec = X_rec[match_idx]

		i = _y_true != _y_pred
		_X_train = X_train[i]
		_X_test = X_test[i]
		_y_pred = _y_pred[i]
		_y_true = _y_true[i]
		_X_rec = _X_rec[i]

		_, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 8))
		for num, axis in enumerate(axes.ravel()):

		    x_test = _X_test[num].copy()
		    test_win = np.argmax(np.cumsum(x_test)) + 1
		    x_test[x_test == 0] = np.nan
		    
		    x_train = _X_train[num].copy()
		    x_train[x_train == 0] = np.nan
		    
		    axis.set_title("Predicted: {}; GT: {}".format(_y_pred[num], _y_true[num]))
		    axis.plot(x_test, "o", label="test")
		    axis.plot(x_train, "o", label="train")
		    axis.plot(_X_rec[num, :test_win])
		    axis.legend()

		plt.tight_layout()

		plt.savefig(path_to_file + "_" + str(combo) + ".pdf")


# TODO: Set zero_thresh using some automated heuristic procedure.
def estimate_zero_thresh():
	pass


def plot_active_basis(combos, Us, Vs, configs, path_to_file, zero_thresh=1e-7):

	combos = np.array(combos)
	max_iters = [configs[combo]["mf_lars_config"]["max_iter"] for combo in combos]
	combos = combos[np.argsort(max_iters)]

	_, axes = plt.subplots(nrows=2, ncols=len(combos) // 2, figsize=(15, 8))
	for axis, combo in zip(axes.ravel(), combos):

		# Set small coefficients to zero.
		U = np.abs(Us[combo])
		U[U < zero_thresh] = 0
		active = np.sum(U, axis=0) != 0

		V = Vs[combo]
		V = V[:, active]
		
		rank = configs[combo]["experiment_config"]["rank"]
		max_iter = configs[combo]["mf_lars_config"]["max_iter"]

		axis.plot(V)
		axis.set_title("Iterations {}; rank: {}".format(max_iter, rank))

	plt.tight_layout()

	plt.savefig(path_to_file + ".pdf")


def plot_weights_U(combos, Us, configs, path_to_file, zero_thresh=1e-7):

	combos = np.array(combos)
	max_iters = [configs[combo]["mf_lars_config"]["max_iter"] for combo in combos]
	combos = combos[np.argsort(max_iters)]

	_, axes = plt.subplots(nrows=2, ncols=len(combos), figsize=(15, 8))
	for i, combo in enumerate(combos):

		# Set small coefficients to zero.
		U = np.abs(Us[combo])
		U[U < zero_thresh] = 0
		
		rank = configs[combo]["experiment_config"]["rank"]
		max_iter = configs[combo]["mf_lars_config"]["max_iter"]

		img = axes[0, i].imshow(U, aspect="auto")
		#axes[0, i].set_title("Sparsity in U (rank: {})".format(rank))
		axes[0, i].set_title("Rank")
		if i == 0:
			axes[0, i].set_ylabel("Females")
		axes[0, i].set_xticks([])
		axes[0, i].set_yticks([])
		plt.colorbar(img, ax=axes[0, i], orientation='horizontal', fraction=.03, 
			pad=0.01, label="abs(U)")

		active = np.sum(U, axis=0) != 0
		U_active = U[:, active]
		for u in U_active.T:
		    axes[1, i].plot(sorted(u))

		#axes[1, i].legend(np.arange(rank)[active])
		axes[1, i].set_title("Female")
		if i == 0:
			axes[1, i].set_ylabel("Basis profile index")

	plt.tight_layout()

	plt.savefig(path_to_file + f"_rank{rank}.pdf")


def plot_sparsity_U(combos, Us, configs, path_to_file, zero_thresh=1e-7):

	fontsize = 14

	combos = np.array(combos)
	max_iters = [configs[combo]["mf_lars_config"]["max_iter"] for combo in combos]
	combos = combos[np.argsort(max_iters)]

	fig, axes = plt.subplots(nrows=3, ncols=len(combos), figsize=(15, 8))
	for i, combo in enumerate(combos):

		# Set small coefficients to zero.
		U = np.abs(Us[combo])
		U[U < zero_thresh] = 0
		
		rank = configs[combo]["experiment_config"]["rank"]
		max_iter = configs[combo]["mf_lars_config"]["max_iter"]

		img = axes[0, i].imshow(U, aspect="auto") 
		axes[0, i].set_title("Sparsity = {:.0f}%  ".format(
			np.sum(U == 0) / U.size * 100), fontsize=fontsize-2)
		if i == 0:
			axes[0, i].set_ylabel("Females", fontsize=fontsize)
		axes[0, i].set_xticks([])
		axes[0, i].set_yticks([])
		plt.colorbar(img, ax=axes[0, i], orientation='horizontal', fraction=.05, 
			pad=0.01, label=r"$\left | \mathbf{U} \right |$")

		img = axes[1, i].spy(U, aspect="auto")
		if i == 0:
			axes[1, i].set_ylabel("Females", fontsize=fontsize)
		axes[1, i].set_xticks([])
		axes[1, i].set_yticks([])
	
		# Fraction of zero coefficients per column.
		v = np.arange(rank) 
		c = np.count_nonzero(U, axis=0) / U.shape[0]

		axes[2, i].bar(v, c)
		axes[2, i].set_xticks([])
		if i == 0:
			axes[2, i].set_ylabel(r"Activations ($u_{i, j} \neq 0$)", fontsize=fontsize)
		axes[2, i].set_xlabel("Basis profile", fontsize=fontsize)
		axes[2, i].set_title("Iterations: {}; Max active: {}".format(
			max_iter, max(np.count_nonzero(U, axis=1))), fontsize=fontsize - 2)

	#fig.suptitle(f"Rank = {rank}", y=1, fontsize=fontsize + 2)
		
	plt.tight_layout()

	plt.savefig(path_to_file + ".pdf")


def plot_loss(combos, configs, path_to_file):

	combos = np.array(combos)
	max_iters = [configs[combo]["mf_lars_config"]["max_iter"] for combo in combos]
	combos = combos[np.argsort(max_iters)]

	_, axes = plt.subplots(nrows=len(combos) // 2, ncols=2, figsize=(15, 8))
	for axis, combo in zip(axes.ravel(), combos):

		rank = configs[combo]["experiment_config"]["rank"]
		max_iter = configs[combo]["mf_lars_config"]["max_iter"]

		axis.plot(configs[combo]["run_config"]["loss"])
		axis.set_title("rank: {}; max_iter: {}".format(rank, max_iter))

		plt.tight_layout()

		plt.savefig(path_to_file + ".pdf")


def plot_distribution(combos, configs, Us, path_to_file, zero_thresh=1e-7):

	fontsize = 14

	combos = np.array(combos)
	max_iters = [configs[combo]["mf_lars_config"]["max_iter"] for combo in combos]
	combos = combos[np.argsort(max_iters)]

	fig, axes = plt.subplots(nrows=2, ncols=len(combos), figsize=(15, 7))
	for i, combo in enumerate(combos):

		rank = configs[combo]["experiment_config"]["rank"]
		max_iter = configs[combo]["mf_lars_config"]["max_iter"]

		# Set small coefficients to zero.
		U_tr = zero_threshold_coefs(Us[combo], thresh=zero_thresh)
		sns.violinplot(x=U_tr.ravel(), ax=axes[0, i])
		axes[0, i].set_title("Sparsity = {:.0f}%".format(
			np.sum(U_tr == 0) / U_tr.size * 100), fontsize=fontsize-2)


	#fig.suptitle(f"Rank = {rank}", y=1, fontsize=fontsize + 2)

	plt.tight_layout()

	plt.savefig(path_to_file + ".pdf")


def plot_distances(combos, configs, Us, Vs, path_to_file, zero_thresh=1e-7):

	fontsize = 14

	combos = np.array(combos)
	max_iters = [configs[combo]["mf_lars_config"]["max_iter"] for combo in combos]
	combos = combos[np.argsort(max_iters)]

	fig, axes = plt.subplots(nrows=2, ncols=len(combos), figsize=(15, 7))
	for i, combo in enumerate(combos):

		rank = configs[combo]["experiment_config"]["rank"]
		max_iter = configs[combo]["mf_lars_config"]["max_iter"]

		U = np.abs(Us[combo])
		U[U < zero_thresh] = 0
		active = np.sum(U, axis=0) != 0

		D = pairwise_distances(U[:, active].T, metric=wasserstein_distance)
		axes[0, i].imshow(D)
		axes[1, i].set_title("max_iter: {}; max num active: {}".format(
			max_iter, max(np.count_nonzero(U, axis=1))), fontsize=fontsize)
		axes[0, i].set_xticks([])
		axes[0, i].set_yticks([])
		if i == 0:
			axes[0, i].set_ylabel("Basis weights", fontsize=fontsize)
			axes[1, i].set_ylabel("Basis profile", fontsize=fontsize)

		axes[0, i].set_xlabel("Basis weights", fontsize=fontsize)
		axes[1, i].set_xlabel("Basis profile", fontsize=fontsize)

		V = Vs[combo]
		D = pairwise_distances(V[:, active].T, metric=wasserstein_distance)
		axes[1, i].imshow(D)
		axes[1, i].set_xticks([])
		axes[1, i].set_yticks([])

	#fig.suptitle("Wasserstein distances (rank={})".format(rank), y=1, fontsize=fontsize)

	plt.tight_layout()

	plt.savefig(path_to_file + ".pdf")


def _kmeans_parameter_selection(rank, U, path_to_file):

	errors = []
	num_clusters = np.arange(4, rank, 3)
	for n_clusters in num_clusters:
	    
	    kmeans = KMeans(n_clusters=n_clusters,
	                    max_iter=300,
	                    random_state=42)
	    kmeans.fit(U)

	    errors.append(kmeans.inertia_)

	plt.figure()
	plt.plot(errors, label=f"Opt. clusters: {num_clusters[np.argmin(errors)]}")
	plt.xticks(np.arange(len(num_clusters)), num_clusters)
	plt.tight_layout()
	plt.ylabel("inertia")
	plt.xlabel("Num Clusters")
	plt.savefig(path_to_file + f"_kmeans_rank{rank}.pdf")

	kmeans = KMeans(n_clusters=num_clusters[np.argmin(errors)],
	                max_iter=300,
	                random_state=42)
	kmeans.fit(U)

	return kmeans


# ERROR:
def kmeans_analysis(combos, configs, Us, Vs, path_to_file):
	
	combos = np.array(combos)
	max_iters = [configs[combo]["mf_lars_config"]["max_iter"] for combo in combos]
	combos = combos[np.argsort(max_iters)]

	fig, axes = plt.subplots(nrows=2, ncols=len(combos), figsize=(15, 7))
	for i, combo in enumerate(combos):

		rank = configs[combo]["experiment_config"]["rank"]
		max_iter = configs[combo]["mf_lars_config"]["max_iter"]

		U = zero_threshold_coefs(Us[combo], thresh=zero_thresh)
		active = np.sum(U, axis=0) != 0

		kmeans = _kmeans_parameter_selection(rank, U, path_to_file)

		sns.violinplot(x=U_tr.ravel(), ax=axes[0, i])
		axes[0, i].set_title("Sparsity = {:.0f}%".format(
			np.sum(U_tr == 0) / U_tr.size * 100), fontsize=fontsize-2)


	#fig.suptitle(f"Rank = {rank}", y=1, fontsize=fontsize + 2)

	plt.tight_layout()

	plt.savefig(path_to_file + ".pdf")


	

def main():
	# NOTE:
	# * Each female should have no more than max_iter number of non-zero coefficients. 
	# * The number of active coefficients is derived for all females and might give
	#   number of non-zero coefficients > max_iter.

	path_to_results = "/Users/sela/Desktop/tsd_code/results/mf_lars_rank_estimate"
	configs, M_hats, y_preds, y_trues, mcc, Us, Vs = get_results(path_to_results)
	# TODO: iterate through ranks.

	# One figure with subplots for each max_iter per rank.
	target_rank = 10

	combos = []
	for combo, results in configs.items(): 
		if results["experiment_config"]["rank"] == target_rank:
			combos.append(combo)

	X_train = np.load("/Users/sela/Desktop/tsd_code/data/screening_filtered/train/X_train.npy")
	X_test = np.load("/Users/sela/Desktop/tsd_code/data/screening_filtered/test/X_test.npy")

	# TODO: 
	# * K-means analysis.
	# * Covariance (sklearn) between profiles and between weights.

	"""
	kmeans_analysis(
		combos=combos,
		configs=configs, 
		Us=Us,
		Vs=Vs,
		path_to_file="/Users/sela/Desktop/mf_lars_rank_estimate/tsd_code_setup/distribution"
	)
	"""

	base_path = "/Users/sela/Desktop/mf_lars_rank_estimate/tsd_code_setup/"

	plot_reconstructed(
		combos=combos,
		M_hats=M_hats,
		X_train=X_train,
		path_to_file=f"{base_path}/reconstructed_rank{target_rank}"
	)

	plot_predictions(
		combos=combos,
		M_hats=M_hats,
		X_test=X_test,
		y_preds=y_preds,
		y_trues=y_trues,
		path_to_file=f"{base_path}/predict_rank{target_rank}"
	)

	plot_loss(
		combos=combos,
		configs=configs, 
		path_to_file=f"{base_path}/loss_rank{target_rank}"
	)

	plot_active_basis(
		combos=combos, 
		Us=Us,
		Vs=Vs,
		configs=configs, 
		path_to_file=f"{base_path}/basis_rank{target_rank}"
	)

	plot_distances(
		combos=combos, 
		Us=Us,
		Vs=Vs,
		configs=configs, 
		path_to_file=f"{base_path}/basis_dist_rank{target_rank}"
	)

	plot_sparsity_U(
		combos=combos, 
		Us=Us,
		configs=configs, 
		path_to_file=f"{base_path}/sparsity_U_rank{target_rank}"
	)

	plot_distribution(
		combos=combos,
		configs=configs, 
		Us=Us,
		path_to_file=f"{base_path}/distribution_rank{target_rank}"
	)


if __name__ == "__main__":
	main()