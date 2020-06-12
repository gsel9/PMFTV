from .base_config import BaseConfig


class GDLConfig(BaseConfig):

	def __init__(self, **kwargs):

		super().__init__()
		
		# Update config holding only defaults.		
		self.config = self.default_settings()
		self.config.update(**kwargs)

		for prop, value in self.config.items():
			setattr(self, prop, value)

	def default_settings(self):

		default = {
			"row_gamma": 1e-10,
			"col_gamma": 1e-10,
			"tv_gamma": None,
			"conv_gamma": None,
			"degree_row_poly": 5,
			"degree_col_poly": 5,
			"diffusion_steps": 10,
			"channels": 32,
			"optimiser": "adam",
			"learning_rate": 1e-3,
			"init_basis": None
		}
		return default.copy()

	def get_config(self):

		return {"gdl_config": self.config}


class RowGraphConfig(BaseConfig):

	def __init__(self, **kwargs):

		super().__init__()
		
		# Update config holding only defaults.		
		self.config = self.default_settings()
		self.config.update(**kwargs)

		for prop, value in self.config.items():
			setattr(self, prop, value)

	def default_settings(self):

		default = {
			"path_distance_matrix": None,
			"path_data_matrix": None,
			"method": None,
			"k": None
		}
		return default.copy()

	def get_config(self):

		return {"row_graph_config": self.config}


class ColumnGraphConfig(BaseConfig):

	def __init__(self, **kwargs):

		super().__init__()
		
		# Update config holding only defaults.
		self.config = self.default_settings()
		self.config.update(**kwargs)

		for prop, value in self.config.items():
			setattr(self, prop, value)

	def default_settings(self):

		default = {
			"method": None,
			"num_nodes": None,
			"weights": None,
			"k": None
		}
		return default.copy()

	def _make_json_serialiseable_serializable(self):

		self.config["weights"] = [int(weight) for weight in self.config["weights"]] 

	def get_config(self):

		self._make_json_serialiseable_serializable()

		return {"col_graph_config": self.config}
