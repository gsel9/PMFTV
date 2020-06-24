from .base_config import BaseConfig


class ExperimentConfig(BaseConfig):

	def __init__(self, **kwargs):

		super().__init__()
		
		# Update config holding only defaults.		
		self.config = self.default_settings()
		self.config.update(**kwargs)

		for prop, value in self.config.items():
			setattr(self, prop, value)

	def default_settings(self):

		# Adjustable parameters.
		default = {
			"exp_id": None,	
			"path_to_results": None,
			"save_only_configs": None,
			"path_data_file": None,
			"num_train_samples": None,
			"num_val_samples": None,
			"rank": None,
			"num_epochs": None,
			"time_lag": None,
			"epochs_per_display": None,
			"epochs_per_val": None,
			"seed": 42,
			"n_kfold_splits": None,
			"domain": None,
			"early_stopping": None,
			"shuffle": False,
			"val_size": 0,
			"patience": 0,
			"chances_to_improve": 0,
			"monitor_loss": True,
			"tol": 1e-5
		}
		return default.copy()

	def update_value(self, key, value):

		self.config[key] = value

	def update_config(self, updates):

		for key, value in updates.items():

			if isinstance(self.config[key], list):
				self.config[key].append(value)

			else:
				self.config[key] = value

			setattr(self, key, value)

	def get_config(self):

		return {"experiment_config": self.config}


class RunConfig(BaseConfig):

	def __init__(self, **kwargs):

		super().__init__()
		
		# Update config holding only defaults.		
		self.config = self.default_settings()
		self.config.update(**kwargs)

		for prop, value in self.config.items():
			setattr(self, prop, value)

		self.model_weights = None 
		self.U = None 
		self.V = None
		self.M_hat = None 
		self.X_true = None 
		self.X_pred = None
		self.O_val = None

	def default_settings(self):

		# Logged parameters.
		default = {
			"mcc": [],
			"sensitivity": [],
			"specificity": [],
			"mcc_binary": [],
			"recall_micro": [],
			"accuracy": [],
			"loss": [],
			"rec_mse": [],
			"duration": [], 
			"opt_epoch": None,
			"unique_train_values": None,
	        "train_distribution": None,
	        "unique_val_values": None,
	        "val_distribution": None
		}
		return default.copy()

	def append_value(self, key, value):

		self.config[key].append(value)

	def append_values(self, data):

		for key, value in data.items():
			self.config[key].append(value)

	def update_config(self, updates):

		for key, value in updates.items():

			if isinstance(self.config[key], list):
				self.config[key].append(value)

			else:
				self.config[key] = value

			setattr(self, key, value)

	def make_json_serializable(self):

		if self.config["val_distribution"] is not None:
			self.config["val_distribution"] = [int(v) for v in self.config["val_distribution"]] 
		
		if self.config["unique_val_values"] is not None:
			self.config["unique_val_values"] = [int(v) for v in self.config["unique_val_values"]] 

		if self.config["train_distribution"] is not None:
			self.config["train_distribution"] = [int(v) for v in self.config["train_distribution"]] 
		
		if self.config["unique_train_values"] is not None:
			self.config["unique_train_values"] = [int(v) for v in self.config["unique_train_values"]] 

	def get_config(self):

		self.make_json_serializable()

		return {"run_config": self.config}
