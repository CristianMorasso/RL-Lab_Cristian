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


def training_loop(env, actor_net, critic_net, updateRule, frequency=10, episodes=100):
    actor_optimizer = tf.keras.optimizers.Adam()
    critic_optimizer = tf.keras.optimizers.Adam()
    rewards_list, reward_queue = [], collections.deque(maxlen=100)
    memory_buffer = []

    for ep in range(episodes):
        # reset the environment and obtain the initial state
        state = env.reset()[0]
        state = state.reshape(-1,4)
        ep_reward = 0
        while True:
            # select the action to perform
            p = actor_net(state).numpy()[0]
            action = np.random.choice(2, p=p)

            # Perform the action, store the data in the memory buffer and update the reward
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state.reshape(-1,4)
            memory_buffer.append([state, action, reward, next_state, terminated])

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
        print(f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f})")

    # Close the enviornment and return the rewards list
    env.close()
    return rewards_list

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