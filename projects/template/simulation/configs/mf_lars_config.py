import numpy as np

from .base_config import BaseConfig


class MFLarsConfig(BaseConfig):

	def __init__(self, R=None, K=None, J=None, **kwargs):

		super().__init__()

		self.R = R
		self.K = K
		self.J = J
		
		# Update config holding only defaults.		
		self.config = self.default_settings()
		self.config.update(**kwargs)

		# Set attributes.
		for prop, value in self.config.items():
			setattr(self, prop, value)

		self.model_type = "MFLars"

	def update_value(self, key, value):

		self.config.update({key: value})

	def default_settings(self):

		default = {
	        "lambda0": 1.0,
	        "lambda1": 1.0,
	        "lambda2": 1.0,
	        "lambda3": 1.0,
	        "max_iter": None,
	        "alphas": None,
	        "init_basis": None
	    }
		return default.copy()

	def make_json_serializable(self):

		if self.config["alphas"] is not None:
			self.config["alphas"] = self.config["alphas"].tolist() 

	def get_config(self):

		self.make_json_serializable()

		return {"mf_lars_config": self.config}
