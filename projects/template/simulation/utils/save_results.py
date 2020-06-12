import json

import numpy as np


def save_to_disk(exp_config, run_config, model_config, row_graph_config=None, col_graph_config=None):

	save_configs(
		exp_config=exp_config, 
		run_config=run_config,
		model_config=model_config, 
		row_graph_config=row_graph_config, 
		col_graph_config=col_graph_config
	)

	if exp_config.save_only_configs:
		return

	if run_config.model_weights is not None:
		save_model_weights(f"{exp_config.path_to_results}/{exp_config.exp_id}", run_config.model_weights)

	if run_config.U is not None:
		save_opt_matrices(f"{exp_config.path_to_results}/{exp_config.exp_id}", run_config.U, "U")

	if run_config.V is not None:
		save_opt_matrices(f"{exp_config.path_to_results}/{exp_config.exp_id}", run_config.V, "V")

	if run_config.M_hat is not None:
		save_opt_matrices(f"{exp_config.path_to_results}/{exp_config.exp_id}", run_config.M_hat, "M_hat")

	if run_config.X_true is not None:
		save_opt_matrices(f"{exp_config.path_to_results}/{exp_config.exp_id}", run_config.X_true, "X_true")

	if run_config.X_pred is not None:
		save_opt_matrices(f"{exp_config.path_to_results}/{exp_config.exp_id}", run_config.X_pred, "X_pred")

	if run_config.O_val is not None:
		save_opt_matrices(f"{exp_config.path_to_results}/{exp_config.exp_id}", run_config.O_val, "O_val")


def save_configs(exp_config=None,
		         run_config=None,
		         model_config=None,
		         row_graph_config=None,
		         col_graph_config=None):

	configs = {}

	if exp_config is not None:
		configs.update(exp_config.get_config())

	if run_config is not None:
		configs.update(run_config.get_config())

	if model_config is not None:
		configs.update(model_config.get_config())

	if row_graph_config is not None:
		configs.update(row_graph_config.get_config())

	if col_graph_config is not None:
		configs.update(col_graph_config.get_config())

	with open(f"{exp_config.path_to_results}/{exp_config.exp_id}_configs.json", 'w') as outfile:
		json.dump(configs, outfile, indent=4)

	print(f"Saved configs in {exp_config.path_to_results}")


def save_model_weights(path_to_file, weights):

	with open(f'{path_to_file}_weights.json', 'w') as outfile:
		json.dump(weights, outfile, indent=4)

	print('Saved model weights in:', path_to_file)


def save_opt_matrices(path_to_file, M_hat, file_name):

	np.save(f"{path_to_file}_{file_name}.npy", M_hat)

	print('Saved reconstructed matrix in:', path_to_file)
