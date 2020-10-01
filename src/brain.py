import tensorflow as tf 
from tensorflow import keras 
import numpy as np


class ActorCritic:
	def __init__(self, input_size=25, actions={'throttle' : 5, 'roll' : 3, 'pitch' : 5, 'yaw' : 5, 'jump':2, 'handbrake':2}):
		self.eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
		self.actions = actions 
		output_size = sum(actions.values())
		
		self.input = keras.layers.Input(shape=(input_size, 1, ))
		self.common = keras.layers.LSTM(128)(self.input)

		self.actors, self.critics = list(), list()
		for action, num_pos in actions.items():
			self.actors.append(keras.layers.Dense(num_pos, activation='softmax')(self.common))
			self.critics.append(keras.layers.Dense(1)(self.common))

		self.model = keras.Model(inputs=self.input, outputs=[self.actors, self.critics])
		self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
		self.loss_fn = keras.losses.Huber()
		self.action_history, self.critic_history = dict(), dict()
		self.reward_history = []
		
		for action in actions.keys():
			self.action_history[action] = list()
			self.critic_history[action] = list()


	def __call__(self, state, reward):
		with tf.GradientTape() as tape:
			self.reward_history.append(reward)
			output = dict()
			# call model
			all_outputs = self.model(state)
			for ind, action in enumerate(self.actions):
				self.action_history[action].append(all_outputs[0][ind][0])
				self.critic_history[action].append(all_outputs[1][ind][0])

			# Get output for bot 
			for action, num_options in self.actions.items():
				action_probs = self.action_history[action][-1]
				output[action] = np.random.choice(num_options, p=np.squeeze(action_probs))

			# learn from last output :)
			returns = []
			discounted_sum = 0 
			for r in self.reward_history[::-1]:
				discounted_sum = r + 0.99 * discounted_sum
				returns.insert(0, discounted_sum)

			# normalize (idk why but someone smarter than me said to)
			returns = np.array(returns)
			returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
			returns = returns.tolist()
			
			actor_ls, critic_ls = list(), list()
			for action in self.actions.keys():
				history = zip(self.action_history[action], self.critic_history[action], returns)
				for prob, value, ret in history:
					diff = ret - value
					actor_ls.append(-prob * diff)

					critic_ls.append(
						self.loss_fn(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
					)
			
			# back propagation 
			actor_ls.extend(critic_ls)
			grads = tape.gradient(actor_ls, self.model.trainable_variables)
			self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

		return output 

 		

	def dream(self):
		# run through all examples 
		# learn without reward 
		# :) ?
		pass