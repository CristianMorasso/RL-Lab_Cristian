import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
import math
import time 


# TODO: implement the following functions as in the previous lessons
# def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ): raise NotImplementedError
# def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ): raise NotImplementedError
# def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99, observation_number=None ): raise NotImplementedError
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ): 
	# Initialize the neural network
	model = Sequential()
	
	model.add(Dense(nNodes, input_dim=nInputs, activation='relu'))
	for _ in range(nLayer - 1):
		model.add(Dense(nNodes, activation='relu'))
	model.add(Dense(nOutputs, activation=last_activation))

	return model

def training_loop(env, actor_net, critic_net, updateRule, frequency=10, episodes=100):
	actor_optimizer = tf.keras.optimizers.Adam()
	critic_optimizer = tf.keras.optimizers.Adam()
	rewards_list, reward_queue = [], collections.deque(maxlen=100)
	ep_length_for_mean = collections.deque(maxlen=100)
	ep_lengths = []
	memory_buffer = []
	counter_terminated = 0

	for ep in range(episodes):
		# reset the environment and obtain the initial state
		state = env.reset()[0]
		state_dim = state.shape[0]
		state = state.reshape(-1,state_dim)
		ep_reward = 0
		ep_length = 0
		done = False
		while not done:
			 
			# select the action to perform
			p = actor_net(state).numpy()[0]
			action = np.random.choice(3, p=p)

			# Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, terminated, truncated, info = env.step(action)
			next_state = next_state.reshape(-1,state_dim)
			memory_buffer.append([state, action, reward, next_state, terminated])
			ep_length += 1
			ep_reward += reward

			# exit condition for the episode
			if terminated: counter_terminated += 1
			done = terminated or truncated

			# update the current state
			state = next_state
		# Perform the actual training
		if ep % frequency == 0:
			updateRule(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer)
			memory_buffer = []
		
		# if ep == 100:
		# 	env = gymnasium.make("Acrobot-v1", render_mode="human" )
		# 	env = OverrideReward(env)
		# Update the reward list to return
		
		reward_queue.append(ep_reward)
		rewards_list.append(np.mean(reward_queue))
		
		ep_length_for_mean.append(ep_length)
		ep_lengths.append(np.mean(ep_length_for_mean))
		print(f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f}), episode length: {ep_length}, (averaged: {np.mean(ep_length_for_mean):5.2f}), terminated: {counter_terminated}")

	# Close the enviornment and return the rewards list
	env.close()
	return rewards_list, ep_lengths

def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

	"""
	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
	and for the critic network (or value function)
	"""
	for _ in range(10):
		np.random.shuffle(memory_buffer)
		memory_buffer = 	np.asarray(memory_buffer)
		states =			np.vstack(memory_buffer[:,0])
		next_states = 		np.vstack(memory_buffer[:,3])
		rewards = memory_buffer[:,2]
		dones = memory_buffer[:,4]

		target = rewards + (1 - dones.astype(int)) * gamma * critic_net(next_states).numpy()[0][0]

		with tf.GradientTape() as critic_tape:
			prediction = critic_net(states)
			objective = tf.math.square(prediction - target)
			grad = critic_tape.gradient(objective, critic_net.trainable_variables)
			critic_optimizer.apply_gradients(zip(grad, critic_net.trainable_variables))

	objectives = []
	with tf.GradientTape() as actor_tape:
	   
		actions = np.array(list(memory_buffer[:, 1]), dtype=int)
		# compute the log-prob of the current trajectory and the objective function
		adv_a = rewards + gamma * critic_net(next_states).numpy().reshape(-1)
		adv_b = critic_net(states).numpy().reshape(-1)
		probs = actor_net(states)
		indices = tf.transpose(tf.stack([tf.range(probs.shape[0]), actions]))
		probs = tf.gather_nd(
			indices=indices,
			params=probs
		)
		objective = tf.math.log(probs) * (adv_a - adv_b)
		objectives.append(tf.reduce_mean(tf.reduce_sum(objective)))

		objective = - tf.math.reduce_mean(objectives)
		grads = actor_tape.gradient(objective, actor_net.trainable_variables)
		actor_optimizer.apply_gradients(zip(grads, actor_net.trainable_variables))

class OverrideReward( gymnasium.wrappers.NormalizeReward ):
	"""
	Gymansium wrapper useful to update the reward function of the environment
	"""
	
	def step(self, action):
		previous_observation = np.array(self.env.state, dtype=np.float32)
		observation, reward, terminated, truncated, info = self.env.step( action )
		goal_alt = - np.cos(previous_observation[0]) - np.cos(previous_observation[1] + previous_observation[0])
		if int(-np.cos(observation[0]) - np.cos(observation[1] + observation[0]) < goal_alt):
			reward = -1
		else:
			reward = 1

		if int(-np.cos(observation[0]) - np.cos(observation[1] + observation[0])) >= 1.0:
			reward = 100

		return observation, reward, terminated, truncated, info

def main(): 
	print( "\n***************************************************" )
	print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
	print( "*                 (DRL in Practice)               *" )
	print( "***************************************************\n" )

	_training_steps = 2500

	# Crete the environment and add the wrapper for the custom reward function
	env = gymnasium.make("Acrobot-v1",) # render_mode="human" )
	env = OverrideReward(env)
		
	# Create the networks and perform the actual training
	observation_space = env.observation_space.shape[0]
	action_space = env.action_space.n
	actor_net = createDNN( observation_space, action_space, nLayer=2, nNodes=32, last_activation='softmax' )
	critic_net = createDNN( observation_space, 1, nLayer=2, nNodes=32, last_activation='linear' )
	start = time.time()
	rewards_training, ep_lengths = training_loop( env, actor_net, critic_net, A2C, frequency=10, episodes=_training_steps  )
	end = time.time()
	# Save the trained neural network
	print(f"Durata: {(end - start)/60:.2f} minuti")

	# Plot the results
	t = np.arange(0, _training_steps)
	plt.plot(t, ep_lengths, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "length", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()