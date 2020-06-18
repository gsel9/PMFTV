import numpy as np

from .base_config import BaseConfig


class MFConvConfig(BaseConfig):

	def __init__(self, R=None, K=None, J=None, **kwargs):

		super().__init__()
		
		# Update config holding only defaults.		
		self.config = self.default_settings()
		self.config.update(**kwargs)

		for prop, value in self.config.items():
			setattr(self, prop, value)

		self.R = R
		self.J = J
		self.K = K

		self.model_type = "MFConv"

	def default_settings(self):

		default = {
	        "lambda0": 1.0,
	        "lambda1": 1.0,
	        "lambda2": 1.0,
	        "lambda3": 1.0,
	        "init_basis": None
	    }
		return default.copy()

	def get_config(self):

		return {"mf_conv_config": self.config}


class WeightedMFConvConfig(BaseConfig):

	def __init__(self, R=None, W=None, K=None, **kwargs):

		super().__init__()
		
		# Update config holding only defaults.		
		self.config = self.default_settings()
		self.config.update(**kwargs)

		for prop, value in self.config.items():
			setattr(self, prop, value)

		self.R = R
		self.W = W
		self.K = K

		self.model_type = "WMFConv"

	def default_settings(self):

		default = {
	        "lambda0": 1.0,
	        "lambda1": 1.0,
	        "lambda2": 1.0,
	        "lambda3": 1.0,
	        "init_basis": None
	    }
		return default.copy()

	def get_config(self):

		return {"weighted_mf_conv_config": self.config}
