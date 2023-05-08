import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


def createDNN( nInputs, nOutputs, nLayer, nNodes ):
	"""
	Function that generates a neural network with the given requirements.

	Args:
		nInputs: number of input nodes
		nOutputs: number of output nodes
		nLayer: number of hidden layers
		nNodes: number nodes in the hidden layers
		
	Returns:
		model: the generated tensorflow model

	"""
	
	# Initialize the neural network
	model = Sequential()
	#
	# YOUR CODE HERE!
	#
	model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) 
	for _ in range(1, nLayer):
		model.add(Dense(nNodes, activation="relu"))
	model.add(Dense(nOutputs, activation="softmax"))
	return model


def training_loop( env, neural_net, updateRule, frequency=10, episodes=100 ):
	"""
	Main loop of the reinforcement learning algorithm. Execute the actions and interact
	with the environment to collect the experience for the training.

	Args:
		env: gymnasium environment for the training
		neural_net: the model to train 
		updateRule: external function for the training of the neural network
		
	Returns:
		averaged_rewards: array with the averaged rewards obtained

	"""

	#TODO: initialize the optimizer 
	optimizer = tf.optimizers.Adam() 
	rewards_list, reward_queue = [], collections.deque( maxlen=100 )
	memory_buffer = []
	memory_buffer_totale = [] 
	for ep in range(episodes):

		#TODO: reset the environment and obtain the initial state
		state = env.reset()[0]
		state=state.reshape(-1,4) #None 
		#state = state.reshape(-1,4)
		ep_reward = 0
		while True:

			#TODO: select the action to perform
			distribution = neural_net(state).numpy()[0]
			action = np.random.choice(2, p = distribution)

			#TODO: Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, done, truncated, _ = env.step(action)
			next_state = next_state.reshape(-1,4)
			memory_buffer.append([state, action, next_state, reward, done])
			#memory_buffer.append( None )
			ep_reward += reward

			# Perform the actual training
			#updateRule( neural_net, memory_buffer, optimizer )

			#TODO: exit condition for the episode
			if done or truncated: break

			#TODO: update the current state
			state = next_state
		memory_buffer_totale.append(memory_buffer)
		memory_buffer=[]
		#TODO: Perform the actual training every 'frequency' episodes
	
		if ep%frequency == 0:
			updateRule( neural_net, memory_buffer_totale, optimizer )
			memory_buffer = []
			memory_buffer_totale = []
		# Update the reward list to return
		reward_queue.append( ep_reward )
		rewards_list.append( np.mean(reward_queue) )
		print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})" )

	# Close the enviornment and return the rewards list
	env.close()
	return rewards_list



def REINFORCE_naive( neural_net, memory_buffer, optimizer ):
	"""
	Main update rule for the REINFORCE process, the naive implementation of the policy-gradient theorem.

	"""
	# np.array(len(memory_buffer))
	#TODO: Setup the tape
	with tf.GradientTape() as tape:
		objectives = []
		#TODO: Initialize the array for the objectives, one for each episode considered
		#TODO: Iterate over all the trajectories considered
		for ep in memory_buffer:
			#TODO: Extract the information from the buffer (for the considered episode)
			ep = np.array(ep)
			states = ep[:,0]
			actions = ep[:,1]
			rewards = ep[:,3]
			log_probs = []
			for i in range(len(actions)):
				probabilities = neural_net(states[i])[0][actions[i]]
				log_probs.append(tf.math.log(probabilities))
			log_prob_sum = tf.math.reduce_sum(log_probs)
			objectives.append(log_prob_sum * sum(rewards))
			
		objective = -tf.math.reduce_mean(objectives)
		grads = tape.gradient(objective, neural_net.trainable_variables)
		optimizer.apply_gradients( zip(grads, neural_net.trainable_variables) )
				#TODO: Compute the log-prob of the current trajectory
				#TODO: Implement the update rule, notice that the REINFORCE objective 
				# is the sum of the logprob (i.e., the probability of the trajectory)
				# multiplied by the sum of the reward

			#TODO: Compute the final final objective to optimize

	#raise NotImplemented


def REINFORCE_rw2go( neural_net, memory_buffer, optimizer ):
	"""
	Main update rule for the REINFORCE process, with the addition of the reward-to-go trick,

	"""

	with tf.GradientTape() as tape:
		objectives = []
		#TODO: Initialize the array for the objectives, one for each episode considered
		#TODO: Iterate over all the trajectories considered
		for ep in memory_buffer:
			ep = np.array(ep)
			#TODO: Extract the information from the buffer (for the considered episode)
			states = ep[:,0]
			actions = ep[:,1]
			rewards = ep[:,3]
			log_probs = []
			sums = []
			for i in range(len(actions)):
				probabilities = neural_net(states[i])[0][actions[i]]
				log_probs.append(tf.math.log(probabilities) * sum(rewards[i:]))
				#sums.append(sum(rewards[i:]))
			log_prob_sum = tf.math.reduce_sum(log_probs)
			objectives.append(log_prob_sum)
			
		objective = -tf.math.reduce_mean(objectives)
		grads = tape.gradient(objective, neural_net.trainable_variables)
		optimizer.apply_gradients( zip(grads, neural_net.trainable_variables) )


def main():
	print( "\n*************************************************" )
	print( "*  Welcome to the ninth lesson of the RL-Lab!   *" )
	print( "*                 (REINFORCE)                   *" )
	print( "*************************************************\n" )

	_training_steps = 1500
	env = gymnasium.make( "CartPole-v1" )

	# Training A)
	neural_net = createDNN( 4, 2, nLayer=2, nNodes=32)
	rewards_naive = training_loop( env, neural_net, REINFORCE_naive, episodes=_training_steps  )
	print()

	# Training B)
	neural_net = createDNN( 4, 2, nLayer=2, nNodes=32)
	rewards_rw2go = training_loop( env, neural_net, REINFORCE_rw2go, episodes=_training_steps  )

	# Plot
	t = np.arange(0, _training_steps)
	plt.plot(t, rewards_naive, label="naive", linewidth=3)
	plt.plot(t, rewards_rw2go, label="reward to go", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "reward", fontsize=16)
	plt.legend() 
	plt.show()


if __name__ == "__main__":
	main()	
