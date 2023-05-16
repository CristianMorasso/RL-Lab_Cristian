import warnings; warnings.filterwarnings("ignore")
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf; import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gymnasium, collections


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
    terminated_len_mean = collections.deque(maxlen=100)
    terminated_len = []
    memory_buffer = []
    terminated_count = 0
    for ep in range(episodes):
        # reset the environment and obtain the initial state
        state = env.reset()[0]
        state_dim = 1#state.shape[0]
        state = np.array(state).reshape(-1,state_dim)
        ep_reward = 0
        
        ep_length = 0
        while True:
             
            # select the action to perform
            p = actor_net(state).numpy()[0]
            action = np.random.choice(env.action_space.n, p=p)

            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state).reshape(-1,state_dim)
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
        rewards_list.append(np.mean(reward_queue))
        terminated_count +=info
        terminated_len_mean.append(info)
        terminated_len.append(np.mean(terminated_len_mean))
        print(f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f}), episode length: {ep_length}, (averaged: {np.mean(terminated_len_mean):5.2f}), count: {np.sum(terminated_count)}")
        # if ep % 100 == 0:
        #     print("stati")
        #     print(actor_net(np.array(range(16)).reshape(-1,1)))
        #     print("valori")
        #     print(critic_net(np.array(range(16)).reshape(-1,1)))
    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list, terminated_len

def A2C(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99):
    """
    Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
    and for the critic network (or value function)

    """
    memory_buffer = np.array(memory_buffer)
    for _ in range(10):
        # Sample batch
        np.random.shuffle(memory_buffer)
        states = np.array(list(memory_buffer[:, 0]), dtype=np.float)[:,0,:]
        rewards = np.array(list(memory_buffer[:, 2]), dtype=np.float)
        next_states = np.array(list(memory_buffer[:, 3]), dtype=np.float)[:,0,:]
        done = np.array(list(memory_buffer[:, 4]), dtype=bool)
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
            # update rule for the critic (value function)


# implement the following class
class OverrideReward( gymnasium.wrappers.NormalizeReward ):
    """
	Gymansium wrapper useful to update the reward function of the environment
"""
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step( action )
        if next_state in [5,7,11,12]:
            reward = 0#-1000
            # terminated  = False
            # truncated = True
            info = False
        elif next_state == 15:
            reward = 100
            # terminated  = True
            # truncated = False
            info = True
        else:
            row = np.floor(next_state/4)
            col = next_state%4
            dist = (3 - row) + (3 - col) 
            reward =  (6-dist) * 0.1
            info = False
            
        return next_state, reward, terminated, truncated, info

    





def main(): 
	print( "\n***************************************************" )
	print( "*  Welcome to the eleventh lesson of the RL-Lab!  *" )
	print( "*                 (DRL in Practice)               *" )
	print( "***************************************************\n" )

	_training_steps = 2500

	# Crete the environment and add the wrapper for the custom reward function
	gymnasium.envs.register(
		id='FrozenLake-v1',
		entry_point='gymnasium.envs.toy_text:FrozenLakeEnv',
		max_episode_steps=100
	)
	env = gymnasium.make("FrozenLake-v1", map_name="4x4", is_slippery=True)#, render_mode="human" )
	env = OverrideReward(env)
		
	# Create the networks and perform the actual training
	observation_space = 1
	action_space = env.action_space.n
	actor_net = createDNN( observation_space, action_space, nLayer=2, nNodes=32, last_activation='softmax' )
	critic_net = createDNN( observation_space, 1, nLayer=2, nNodes=32, last_activation='linear' )
	rewards_training, ep_lengths = training_loop( env, actor_net, critic_net, A2C, frequency=5, episodes=_training_steps  )
	# Save the trained neural network
	actor_net.save( "MountainCarActor.h5" )

	# Plot the results
	t = np.arange(0, _training_steps)
	plt.plot(t, ep_lengths, label="A2C", linewidth=3)
	plt.xlabel( "epsiodes", fontsize=16)
	plt.ylabel( "length", fontsize=16)
	plt.legend()
	plt.show()


if __name__ == "__main__":
	main()	