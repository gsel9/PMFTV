import numpy as np 


class Direct1DTV:

	def __init__(self, lam=0.001):

		self.lam = lam
		self.neg_lam = -1.0 * lam

		self.k = 0
		self.k0 = 0
		self.km = 0
		self.kp = 0

		self.v_min = None 
		self.v_max = None 

		self.u_min = lam
		self.u_max = -1.0 * lam

	def bound_shift_down(self, x, y):

		while self.k0 <= self.km:
			x[self.k0] = self.v_min
			self.k0 += 1
		
		self.k = self.k0
		self.km = self.k0
		self.v_min = y[self.k]
		self.u_min = self.lam
		self.u_max = y[self.k] + self.lam - self.v_max

	def bound_shift_up(self, x, y):

		while self.k0 <= self.kp:
			x[self.k0] = self.v_max 
			self.k0 += 1

		self.k = self.k0
		self.kp = self.k0
		self.v_max = y[self.k]
		self.u_max = self.neg_lam
		self.u_min = y[self.k] + self.neg_lam - self.v_min

	def run(self, y):

		x = np.zeros_like(y, dtype=float)

		self.v_min = y[0] - self.lam
		self.v_max = y[0] + self.lam

		while True:

			while self.k == len(y) - 1:

				if self.u_min < 0:
					self.bound_shift_down(x, y)
					
				elif self.u_max > 0:
					self.bound_shift_up(x, y)

				else:
					self.v_min += self.u_min / (self.k - self.k0 + 1)

					while self.k0 <= self.k:
						x[self.k0] = self.v_min
						self.k0 += 1

					return x

			if y[self.k + 1] + self.u_min < self.v_min + self.neg_lam:
				self.shift_down(x, y)

			elif y[self.k + 1] + self.u_max > self.v_max + self.lam:
				self.shift_up(x, y)

			else:

				self.k += 1

				self.u_min += y[self.k] - self.v_min
				self.u_max += y[self.k] - self.v_max

				self.check_bounds(y)

	def shift_down(self, x, y):

		while self.k0 <= self.km:
			x[self.k0] = self.v_min
			self.k0 += 1

		self.k = self.k0
		self.km = self.k0
		self.kp = self.k0
		self.v_min = y[self.k] 
		self.v_max = y[self.k] + 2 * self.lam
		self.u_min = self.lam
		self.u_max = self.neg_lam

	def shift_up(self, x, y):

		while self.k0 <= self.kp:
			x[self.k0] = self.v_max
			self.k0 += 1

		self.k = self.k0
		self.km = self.k0
		self.kp = self.k0
		self.v_min = y[self.k] - 2 * self.lam
		self.v_max = y[self.k]
		self.u_min = self.lam
		self.u_max = self.neg_lam

	def check_bounds(self, y):
					
		if self.u_min >= self.lam:
			self.v_min += (self.u_min - self.lam) / (self.k - self.k0 + 1)
			self.u_min = self.lam
			self.km = self.k

		if self.u_max <= self.neg_lam:
			self.v_max += (self.u_max + self.lam) / (self.k - self.k0 + 1)
			self.u_max = self.neg_lam
			self.kp = self.k


if __name__ == "__main__":
	"""
	Idea is to run this alg on each basis function in V.
	"""
	"""
	import matplotlib.pyplot as plt 

	np.random.seed(42)
	y = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1] 
	y_noisy = y + np.random.random(len(y))

	denoise = Direct1DTV(lam=0.5)
	x = denoise.run(y)
	print(np.round(x))
	print(y)
	"""

	import matplotlib.pyplot as plt
	from skimage import img_as_float
	from skimage.data import astronaut

	lena = img_as_float(astronaut())
	lena = lena[220:300, 220:320]

	noisy = lena + 0.6 * lena.std() * np.random.random(lena.shape)
	noisy = np.clip(noisy, 0, 1)

	denoise = Direct1DTV(lam=0.08)
	denoised = denoise.run(noisy.ravel()).reshape(noisy.shape)

	fig, ax = plt.subplots(ncols=2, figsize=(8, 5))

	plt.gray()
	ax[0].imshow(noisy)
	ax[0].axis('off')
	ax[0].set_title('noisy')
	ax[1].imshow(denoised)
	ax[1].axis('off')
	ax[1].set_title('TV')
	plt.show()

	