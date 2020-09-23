import numpy as np
	
class SVM:

	def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=100):
		self.lr = learning_rate
		self.lambda_param = lambda_param
		self.n_iters = n_iters
		self.w = None
		self.b = None 


	def predict(self, X):
		lin_o = np.dot(X, self.w) - self.b
		return np.sign(lin_o) 

	def fit(self, X, y):
		try:
			with np.load("model.npz") as file:
				self.w = file['w']
				self.b = file['b']
		except Exception as e: 
			y_ = np.where(y <= 0, -1, 1)
			n_samples, n_features = X.shape

			if len(X) != len(y_):
				print("the lengths do not match")
			
			print(y_)
			self.w = np.zeros(n_features)
			self.b = 0

			for _ in range(self.n_iters):
				if _ % 100 == 0:
					print("Training step {}".format(_))
				for i, x_i in enumerate(X):

					condition = y_[i] * np.dot(x_i, self.w) - self.b >= 1
					if condition:
						self.w -= self.lr * 2 * self.lambda_param * self.w 
					else:
						self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[i]))
						self.b -= self.lr * y_[i]
			np.savez("model.npz", w=self.w, b=self.b)


