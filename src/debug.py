from network import *
import numpy as np
import random
import pickle


def main():
	n = Network([1,1])

	f = np.vectorize(lambda X: 0.5 * X)
	print(n,"\n")
	print(n(0.4))
	print(f(0.4),"\n")

	data = np.random.rand(10000)
	n.train(data, f(data), batch_size=50, num_epochs=1000000)


	print(n,"\n")

	print(n(0.4))
	print(f(0.4),"\n")

if __name__ == '__main__':
	main()