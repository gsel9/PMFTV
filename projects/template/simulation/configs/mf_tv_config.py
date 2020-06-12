import numpy as np

from .base_config import BaseConfig


class MFTVConfig(BaseConfig):

	def __init__(self, R=None, J=None, **kwargs):

		super().__init__()
		
		# Update config holding only defaults.		
		self.config = self.default_settings()
		self.config.update(**kwargs)

		for prop, value in self.config.items():
			setattr(self, prop, value)

		self.R = R
		self.J = J

		self.model_type = "MFTV"

	def default_settings(self):

		default = {
	        "lambda0": 1.0,
	        "lambda1": 1.0,
	        "lambda2": 1.0,
	        "lambda3": 1.0,
	        "num_iter": None,
	        "init_basis": None,
	        "gamma": None
	    }
		return default.copy()

	def get_config(self):

		return {"mf_tv_config": self.config}
