from .base_config import BaseConfig


class GDLConfig(BaseConfig):

	def __init__(self, row_graph_config, col_graph_config, **kwargs):

		super().__init__()

		# Update config holding only defaults.		
		self.config = self.default_settings()
		self.config.update(**kwargs)

		for prop, value in self.config.items():
			setattr(self, prop, value)

		self.row_graph_config = self.row_graph_default()
		self.row_graph_config.update(**row_graph_config)

		for prop, value in self.row_graph_config.items():
			setattr(self, prop, value)

		self.col_graph_config = self.col_graph_default()
		self.col_graph_config.update(**col_graph_config)

		for prop, value in self.col_graph_config.items():
			setattr(self, prop, value)

		self.model_type = "GDL"

	def update_value(self, key, value):

		self.config.update({key: value})

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

	def col_graph_default(self):

		default = {
			"method": None,
			"num_nodes": None,
			"weights": None,
			"k": None
		}

		return default.copy()

	def row_graph_default(self):

		default = {
			"path_distance_matrix": None,
			"path_data_matrix": None,
			"method": None,
			"k": None
		}

		return default.copy()

	def get_config(self):

		return {
			"gdl_config": self.config,
			"col_graph_config": self.col_graph_config,
			"row_graph_config": self.row_graph_config
		}

	def _make_json_serialiseable(self):

		self.row_graph_config["weights"] = [int(weight) for weight in self.config["weights"]] 
