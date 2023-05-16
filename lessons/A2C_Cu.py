import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections
import cupy as cp


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

    for ep in range(episodes):
        # reset the environment and obtain the initial state
        state = env.reset()[0]
        state_dim = state.shape[0]
        state = state.reshape(-1,state_dim)
        ep_reward = 0
        
        ep_length = 0
        while True:
             
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
            if terminated or truncated:
                break

            # update the current state
            state = next_state
        # Perform the actual training
        if ep % frequency == 0:
            updateRule(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer)
            memory_buffer = []

        # Update the reward list to return
		
        reward_queue.append(ep_reward)
        rewards_list.append(cp.mean(reward_queue))
        
        ep_length_for_mean.append(ep_length)
        ep_lengths.append(cp.mean(ep_length_for_mean))
        print(f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {cp.mean(reward_queue):5.2f}), episode length: {ep_length}, (averaged: {cp.mean(ep_length_for_mean):5.2f})")
        if ep > 1000:
            env = gymnasium.make( "MountainCarMyVersion-v0", render_mode="human" )
            env = OverrideReward(env)
    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list, ep_lengths

def A2C(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99):
    """
    Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
    and for the critic network (or value function)

    """
    memory_buffer = np.array(memory_buffer)
    for _ in range(10):
        # Sample batch
        np.random.shuffle(memory_buffer)
        states = cp.array(list(memory_buffer[:, 0]), dtype=cp.float)[:,0,:]
        rewards = cp.array(list(memory_buffer[:, 2]), dtype=cp.float)
        next_states = cp.array(list(memory_buffer[:, 3]), dtype=cp.float)[:,0,:]
        done = cp.array(list(memory_buffer[:, 4]), dtype=bool)
        # Tape for the critic
        with tf.GradientTape() as critic_tape:
            # Compute the target and the MSE between the current prediction
            target = rewards + (1 - done.astype(int)) * gamma * critic_net(next_states)
            prediction = critic_net(states)
            objective = tf.math.square(prediction - target)
            grads = critic_tape.gradient(objective, critic_net.trainable_variables)
            critic_optimizer.apply_gradients(zip(grads, critic_net.trainable_variables))
    # implement the update rule for the actor (policy function)
    # extract the information from the buffer for the policy update
    # Tape for the actor
    objectives = []
    with tf.GradientTape() as actor_tape:
       
        actions = cp.array(list(memory_buffer[:, 1]), dtype=int)
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
            # update rule for the critic (value function)


# implement the following class
class OverrideReward( gymnasium.wrappers.NormalizeReward ):
	"""
	Gymansium wrapper useful to update the reward function of the environment
	"""
	# def step(self, action):
	# 	previous_observation = cp.array(self.env.state, dtype=cp.float32)
	# 	observation, reward, terminated, truncated, info = self.env.step( action )
		
	# 	previous_position, previous_velocity = previous_observation
	# 	position, velocity = observation
	# 	shaping_current = position + 0.5
	# 	shaping_previous = previous_position + 0.5
		
	# 	if position >= 0.5:
	# 		reward = 100
	# 	else:
	# 		reward = (shaping_current - shaping_previous) * abs(velocity) * 10000
	# 	return observation, reward, terminated, truncated, info

	def step(self, action):
		previous_observation = cp.array(self.env.state, dtype=cp.float32)
		observation, reward, terminated, truncated, info = self.env.step( action )
		
		position, velocity = observation[0], observation[1]

		if position - previous_observation[0] > 0 and action == 2: reward = 1
		if position - previous_observation[0] > 0 and action == 0: reward = 1
		if position >= 0.5: reward = 100
	
		return observation, reward, terminated, truncated, info

    #test classico
	# def step(self, action):
	# 	previous_observation = cp.array(self.env.state, dtype=cp.float32)
	# 	observation, reward, terminated, truncated, info = self.env.step( action )
		
	# 	position, velocity = observation[0], observation[1]

	# 	if position - previous_observation[0] > 0 and action == 2: reward = 1 #+ abs(0.5 - position)
	# 	if velocity == 0: reward = 0
	# 	if position - previous_observation[0] < 0 and action == 0: reward = 1
	# 	if position >= 0.5: reward = 100
	
	# 	# if velocity > 0 and action == 0: reward = 1
	# 	# if velocity < 0 and action == 2: reward = 1
	# 	# if position >= 0.5 or terminated: reward = 100

	# 	return observation, reward, terminated, truncated, info





def main(): 
	print( "\n***************************************************" )
	print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
	print( "*                 (DRL in Practice)               *" )
	print( "***************************************************\n" )

	_training_steps = 2500

	# Crete the environment and add the wrapper for the custom reward function
	gymnasium.envs.register(
		id='MountainCarMyVersion-v0',
		entry_point='gymnasium.envs.classic_control:MountainCarEnv',
		max_episode_steps=1000
	)
	env = gymnasium.make( "MountainCarMyVersion-v0")#, render_mode="human" )
	env = OverrideReward(env)
		
	# Create the networks and perform the actual training
	observation_space = env.observation_space.shape[0]
	action_space = env.action_space.n
	actor_net = createDNN( observation_space, action_space, nLayer=2, nNodes=32, last_activation='softmax' )
	critic_net = createDNN( observation_space, 1, nLayer=2, nNodes=32, last_activation='linear' )
	rewards_training, ep_lengths = training_loop( env, actor_net, critic_net, A2C, frequency=1, episodes=_training_steps  )
	# Save the trained neural network
	actor_net.save( "MountainCarActor.h5" )

	# Plot the results
	t = cp.arange(0, _training_steps)
	plt.plot(t, ep_lengths, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "length", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	