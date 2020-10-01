from brain import *
import numpy as np
import tensorflow as tf
import random
import pickle


def main():
	n = ActorCritic()
	test_state = tf.random.uniform((25,1,))
	n(test_state, 100)

if __name__ == '__main__':
	main()