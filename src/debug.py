from network import *
import numpy as np
import random
import pickle


def main():
	with open("hello.nn", 'rb') as f:
		n = pickle.load(f)
		
	f = np.vectorize(lambda X: 2 * X)
	
	print(n(0.4))
	print(f(0.4),"\n")
	print(n,"\n")


if __name__ == '__main__':
	main()