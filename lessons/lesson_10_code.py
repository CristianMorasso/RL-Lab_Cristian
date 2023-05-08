import time
import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
from tqdm import tqdm 

# TODO: implement the following functions as in the previous lessons
# Notice that the value function has only one output with a linear activation
# function in the last layer
def createDNN( nInputs, nOutputs, nLayer, nNodes, last_activation ):
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
	
	model.add(Dense(nNodes, input_dim=nInputs, activation="relu")) 
	for _ in range(nLayer-1):
		model.add(Dense(nNodes, activation="relu"))
	model.add(Dense(nOutputs, activation=last_activation))
	return model


def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ):
	
	#TODO: initialize the optimizer 
	iter = 10
	actor_optimizer = tf.optimizers.Adam()
	critic_optimizer = tf.optimizers.Adam() 
	rewards_list, reward_queue = [], collections.deque( maxlen=100 )
	memory_buffer = []
	memory_buffer_totale = [] 
	memory_buffer_no_eps = []
	last_time = time.time()
	#for ep in tqdm(range(episodes)):
	for ep in range(episodes):
		#TODO: reset the environment and obtain the initial state
		state = env.reset()[0]
		#state=state.reshape(-1,4) #None 
		#state = state.reshape(-1,4)
		ep_reward = 0
		#memory_buffer_no_eps = []
		while True:

			#TODO: select the action to perform
			distribution = actor_net(state.reshape(-1,4)).numpy()[0]
			action = np.random.choice(2, p = distribution)

			#TODO: Perform the action, store the data in the memory buffer and update the reward
			next_state, reward, done, truncated, _ = env.step(action)
			#next_state = next_state.reshape(-1,4)
			
			memory_buffer.append([state, action, next_state, reward, done])
			memory_buffer_no_eps.append([state, action, next_state, reward, done])
			#memory_buffer.append( None )
			ep_reward += reward

			# Perform the actual training
			#updateRule( neural_net, memory_buffer, optimizer )

			#TODO: exit condition for the episode
			if done or truncated: break

			#TODO: update the current state
			state = next_state
		memory_buffer_totale.append(memory_buffer)
		memory_buffer = []
		#TODO: Perform the actual training every 'frequency' episodes
		if ep%frequency == 0:
			updateRule(actor_net, critic_net, memory_buffer_totale,memory_buffer_no_eps, actor_optimizer, critic_optimizer )
			#memory_buffer = []
			memory_buffer_no_eps = []
			memory_buffer_totale = []
		# Update the reward list to return
		reward_queue.append( ep_reward )
		rewards_list.append( np.mean(reward_queue) )
		if ep%iter == 0:
			print(f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f}), time: {round(iter/(time.time() -last_time),2)}iter/sec" )
			last_time = time.time()

	# Close the enviornment and return the rewards list
	env.close()
	return rewards_list

def A2C( actor_net, critic_net, memory_buffer,memory_buffer_no_eps, actor_optimizer, critic_optimizer, gamma=0.99 ):

	"""
	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
	and for the critic network (or value function)

	"""
	
	#TODO: implement the update rule for the critic (value function)
	#memory_buffer_no_eps = np.reshape(memory_buffer, (-1,5))
	memory_buffer_no_eps = np.array(memory_buffer_no_eps)#[0]
	for _ in range(10):
		# Shuffle the memory buffer
		np.random.shuffle( memory_buffer_no_eps )
		#TODO: extract the information from the buffer
		states = memory_buffer_no_eps[:,0]
		next_states = memory_buffer_no_eps[:,2]
		rewards = memory_buffer_no_eps[:,3]
		dones = memory_buffer_no_eps[:,4]
		for i in range(len(rewards)): 
			target = rewards[i] + (1-int(dones[i])) * gamma * critic_net(next_states[i].reshape(-1,4))#.numpy()[0]
		
		# Tape for the critic
			with tf.GradientTape() as critic_tape:
				prediction = critic_net(states[i].reshape(-1,4))
				objective = tf.math.square(prediction - target.numpy())
				grads = critic_tape.gradient(objective, critic_net.trainable_variables)
				critic_optimizer.apply_gradients( zip(grads, critic_net.trainable_variables) )

			#TODO: Compute the target and the MSE between the current prediction
			# and the expected advantage 
			#TODO: Perform the actual gradient-descent process
			# raise NotImplementedError

	#TODO: implement the update rule for the actor (policy function)
	#TODO: extract the information from the buffer for the policy update
	# Tape for the actor
	objectives = []
	with tf.GradientTape() as actor_tape:
		
		for ep in memory_buffer:
			ep = np.array(ep)
			#TODO: Extract the information from the buffer (for the considered episode)
			states = ep[:,0]
			next_states = ep[:,2]
			actions = ep[:,1]
			rewards = ep[:,3]
			log_probs = []
			
			#TODO: compute the log-prob of the current trajectory and 
			# the objective function, notice that:
			# the REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory)
			# multiplied by advantage
			adv = []
			for i in range(len(actions)):
				probabilities = actor_net(states[i].reshape(-1,4))[0][actions[i]]
				
				adv_a = rewards[i] + gamma * (critic_net(next_states[i].reshape(-1,4)).numpy()).reshape(-1)
				adv_b = (critic_net(states[i].reshape(-1,4)).numpy()).reshape(-1)
				
				log_probs.append(tf.math.log(probabilities) * (adv_a - adv_b))
			log_prob_sum = tf.math.reduce_sum(log_probs)
			objectives.append(log_prob_sum)
		
		objective = -tf.math.reduce_mean(objectives)
		grads = actor_tape.gradient(objective, actor_net.trainable_variables)
		actor_optimizer.apply_gradients( zip(grads, actor_net.trainable_variables) )

		#TODO: compute the final objective to optimize, is the average between all the considered trajectories
		#raise NotImplementedError
	

def main(): 
	print( "\n*************************************************" )
	print( "*  Welcome to the tenth lesson of the RL-Lab!   *" )
	print( "*                    (A2C)                      *" )
	print( "*************************************************\n" )

	_training_steps = 2500

	env = gymnasium.make( "CartPole-v1" )
	actor_net = createDNN( 4, 2, nLayer=2, nNodes=32, last_activation="softmax")
	critic_net = createDNN( 4, 1, nLayer=2, nNodes=32, last_activation="linear")
	rewards_naive = training_loop( env, actor_net, critic_net, A2C, episodes=_training_steps  )

	t = np.arange(0, _training_steps)
	plt.plot(t, rewards_naive, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "reward", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	
