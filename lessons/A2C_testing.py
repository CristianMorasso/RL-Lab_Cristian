#S
# def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ): 

# 	critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# 	actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# 	rewards_list, reward_queue = [], collections.deque( maxlen=100 )
# 	ep_lengths = []
# 	memory_buffer = []
# 	for ep in range(episodes):
# 		state = env.reset()[0]
# 		state = state.reshape(-1, 4)
# 		ep_reward = 0
# 		ep_length = 0
# 		done = False
# 		while not done:
# 			distribution = actor_net(state.reshape(-1, 4)).numpy()[0]
# 			action = np.random.choice(2, p=distribution)
# 			next_state, reward, terminated, truncated, _ = env.step(action)
# 			next_state = next_state.reshape(-1, 4)
# 			memory_buffer.append([state, action, reward, next_state, terminated])
# 			ep_reward += reward
# 			ep_length += 1

# 			done = terminated or truncated

# 			state = next_state

# 		if ep % frequency == 0:
# 			updateRule(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99)
# 			memory_buffer = []
		

# 		reward_queue.append( ep_reward )
# 		rewards_list.append( np.mean(reward_queue) )
# 		ep_lengths.append(ep_length)
# 		print( f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f}), episode length: {ep_length}" )

# 	# Close the enviornment and return the rewards list
# 	env.close()
# 	return rewards_list#, ep_lengths


# #S L 
# def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

# 	"""
# 	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
# 	and for the critic network (or value function)
# 	"""
# 	for _ in range(10):
# 		np.random.shuffle(memory_buffer)
# 		memory_buffer = np.asarray(memory_buffer)
# 		states =			np.vstack(memory_buffer[:,0])
# 		rewards = 			np.vstack(memory_buffer[:,2])
# 		next_states = 		np.vstack(memory_buffer[:,3])
# 		dones = 			np.vstack(memory_buffer[:,4])
# 		dones = 			np.vstack(dones)

# 		target = rewards + (1 - dones.astype(int)) * gamma * critic_net(next_states).numpy()[0][0]

# 		with tf.GradientTape() as critic_tape:
# 			prediction = critic_net(states)
# 			objective = tf.math.square(prediction - target)
# 			grad = critic_tape.gradient(objective, critic_net.trainable_variables)
# 			critic_optimizer.apply_gradients(zip(grad, critic_net.trainable_variables))


# 	with tf.GradientTape() as actor_tape:
# 		objectives = []
# 		#actions = np.vstack(memory_buffer[:,1])
# 		actions = np.array(list(memory_buffer[:, 1]), dtype=int)
# 		target = 0
# 		probabilities = actor_net(states)
# 		#probability = [x[actions[i][0]] for i,x in enumerate(probabilities)]

# 		indices = tf.transpose(tf.stack([tf.range(probabilities.shape[0]), actions]))
# 		probability = tf.gather_nd(
#             indices=indices,
#             params=probabilities
#         )

# 		log_prob = tf.math.log(probability)
# 		adv_a = rewards + (1 - dones.astype(int)) * gamma * critic_net(next_states).numpy().reshape(-1)
# 		adv_b = critic_net(states).numpy().reshape(-1)
# 		target += log_prob * (adv_a[0] - adv_b[0])
# 		objectives.append(target)

				
# 		objective = -tf.math.reduce_mean(objectives)
# 		grad = actor_tape.gradient(objective, actor_net.trainable_variables)
# 		actor_optimizer.apply_gradients(zip(grad, actor_net.trainable_variables))

#MIO
# def training_loop( env, actor_net, critic_net, updateRule, frequency=10, episodes=100 ):
	
# 	#TODO: initialize the optimizer 
# 	iter = 10
# 	actor_optimizer = tf.optimizers.Adam()
# 	critic_optimizer = tf.optimizers.Adam() 
# 	rewards_list, reward_queue = [], collections.deque( maxlen=100 )
# 	memory_buffer = []
# 	memory_buffer_totale = [] 

# 	last_time = time.time()
# 	#for ep in tqdm(range(episodes)):
# 	for ep in range(episodes):
# 		#TODO: reset the environment and obtain the initial state
# 		state = env.reset()[0]
# 		#state=state.reshape(-1,4) #None 
# 		#state = state.reshape(-1,4)
# 		ep_reward = 0
# 		#memory_buffer_no_eps = []
# 		while True:

# 			#TODO: select the action to perform
# 			distribution = actor_net(state.reshape(-1,4)).numpy()[0]
# 			action = np.random.choice(2, p = distribution)

# 			#TODO: Perform the action, store the data in the memory buffer and update the reward
# 			next_state, reward, done, truncated, _ = env.step(action)
# 			#next_state = next_state.reshape(-1,4)
			
# 			memory_buffer.append([state, action, next_state, reward, done])
# 			#memory_buffer_no_eps.append([state, action, next_state, reward, done])
# 			#memory_buffer.append( None )
# 			ep_reward += reward

# 			# Perform the actual training
# 			#updateRule( neural_net, memory_buffer, optimizer )

# 			#TODO: exit condition for the episode
# 			if done or truncated: break

# 			#TODO: update the current state
# 			state = next_state
# 		#memory_buffer_totale.append(memory_buffer)
# 		#memory_buffer = []
# 		#TODO: Perform the actual training every 'frequency' episodes
# 		if ep%frequency == 0:
# 			updateRule(actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer )
# 			memory_buffer = []
			
# 			#memory_buffer_totale = []
# 		# Update the reward list to return
# 		reward_queue.append( ep_reward )
# 		rewards_list.append( np.mean(reward_queue) )
# 		if ep%iter == 0:
# 			print(f"episode {ep:4d}: rw: {int(ep_reward):3d} (averaged: {np.mean(reward_queue):5.2f}), time: {round(iter/(time.time() -last_time),2)}iter/sec" )
# 			last_time = time.time()

# 	# Close the enviornment and return the rewards list
# 	env.close()
# 	return rewards_list

# def A2C( actor_net, critic_net, memory_buffer, actor_optimizer, critic_optimizer, gamma=0.99 ):

# 	"""
# 	Main update rule for the A2C update. This function includes the updates for the actor network (or policy function)
# 	and for the critic network (or value function)

# 	"""
	
# 	#TODO: implement the update rule for the critic (value function)
	
# 	for _ in range(10):
		
# 		np.random.shuffle( memory_buffer )
# 		ep = np.array(memory_buffer)
# 		#for ep in memory_buffer:
# 			# ep = np.array(ep)
# 			# np.random.shuffle( ep )
# 		# Shuffle the memory buffer
			
# 		#TODO: extract the information from the buffer
# 		states = np.array(list(ep[:,0]), dtype=np.float)
# 		next_states = np.array(list(ep[:,2]), dtype=np.float)
# 		rewards = np.array(list(ep[:,3]), dtype=np.float)
# 		dones = np.array(list(ep[:,4]), dtype=np.float)
# 		# for ist in ep:
# 		# 	target = ist[3] + (1-int(ist[4])) * gamma * critic_net(ist[2].reshape(-1,4)).numpy()
		
# 		target = rewards + (1-dones.astype(int)) * gamma * critic_net(next_states).numpy().reshape(-1)
# 		#print(target.shape)
# 			# Tape for the critic
# 		with tf.GradientTape() as critic_tape:
# 			prediction = critic_net(states.reshape(-1,4))
# 			mse = tf.math.square(prediction - target)
# 			grads = critic_tape.gradient(mse, critic_net.trainable_variables)
# 			critic_optimizer.apply_gradients( zip(grads, critic_net.trainable_variables) )



# 	#memory_buffer_no_eps = np.array(memory_buffer)#[0]
# 	# for _ in range(10):
# 	# 	for memory_buffer_no_eps in memory_buffer:
# 	# 		memory_buffer_no_eps = np.array(memory_buffer_no_eps)
# 	# 		# Shuffle the memory buffer
# 	# 		np.random.shuffle( memory_buffer_no_eps )
# 	# 		#TODO: extract the information from the buffer
# 	# 		states = memory_buffer_no_eps[:,0]
# 	# 		next_states = memory_buffer_no_eps[:,2]
# 	# 		rewards = memory_buffer_no_eps[:,3]
# 	# 		dones = memory_buffer_no_eps[:,4]
# 	# 		for i in range(len(rewards)): 
# 	# 			target = rewards[i] + (1-int(dones[i])) * gamma * critic_net(next_states[i].reshape(-1,4))#.numpy()[0]
			
# 	# 		# Tape for the critic
# 	# 			with tf.GradientTape() as critic_tape:
# 	# 				prediction = critic_net(states[i].reshape(-1,4))
# 	# 				objective = tf.math.square(prediction - target.numpy())
# 	# 				grads = critic_tape.gradient(objective, critic_net.trainable_variables)
# 	# 				critic_optimizer.apply_gradients( zip(grads, critic_net.trainable_variables) )

# 	#TODO: implement the update rule for the actor (policy function)
# 	#TODO: extract the information from the buffer for the policy update
# 	# Tape for the actor
# 	objectives = []
# 	with tf.GradientTape() as actor_tape:
		
# 		#for ep in memory_buffer:
# 		# ep = np.array(ep)
# 		# #TODO: Extract the information from the buffer (for the considered episode)
# 		states = np.array(list(ep[:,0]), dtype=np.float)
# 		actions = np.array(list(ep[:,1]), dtype=np.int)
# 		next_states = np.array(list(ep[:,2]), dtype=np.float)
# 		rewards = np.array(list(ep[:,3]), dtype=np.float)
# 		dones = np.array(list(ep[:,4]), dtype=np.float)
		
# 		# log_probs = []
		
# 		# TODO: compute the log-prob of the current trajectory and 
# 		# the objective function, notice that:
# 		# the REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory)
# 		# multiplied by advantage
# 		probs = []
# 		# for i in range(len(actions)):
# 		probabilities = actor_net(states)
# 		for i,p in enumerate(probabilities):
# 			probs.append(p[actions[i]])
# 		# probs = tf.gather_nd(
#         #         indices=actions,
#         #         params=probabilities
#         #     )
		
# 		adv_a = rewards + gamma * (critic_net(next_states).numpy()).reshape(-1)
# 		adv_b = (critic_net(states).numpy()).reshape(-1)
		
# 		#log_probs.append(tf.math.log(probs) * (adv_a - adv_b))
# 		log_prob_sum = tf.math.reduce_sum(tf.math.log(probs) * (adv_a - adv_b))
# 		#objectives.append(log_prob_sum)
		
# 		# objective = -log_prob_sum

# 		#for ep in memory_buffer:
# 		# ep = np.array(memory_buffer)
# 		# #TODO: Extract the information from the buffer (for the considered episode)
# 		# states = np.array(list(ep[:,0]), dtype=np.float)
# 		# actions = np.array(list(ep[:,1]), dtype=np.float)
# 		# next_states = np.array(list(ep[:,2]), dtype=np.float)
# 		# rewards = np.array(list(ep[:,3]), dtype=np.float)
# 		# dones = np.array(list(ep[:,4]), dtype=np.float)
		
# 		# log_probs = []
		
# 		# #TODO: compute the log-prob of the current trajectory and 
# 		# # the objective function, notice that:
# 		# # the REINFORCE objective is the sum of the logprob (i.e., the probability of the trajectory)
# 		# # multiplied by advantage
# 		# adv = []
# 		# for i in range(len(actions)):
# 		# 	probabilities = actor_net(states[i].reshape(-1,4))[0][int(actions[i])]
			
# 		# 	adv_a = rewards[i] + gamma * (critic_net(next_states[i].reshape(-1,4)).numpy()).reshape(-1)
# 		# 	adv_b = (critic_net(states[i].reshape(-1,4)).numpy()).reshape(-1)
			
# 		# 	log_probs.append(tf.math.log(probabilities) * (adv_a - adv_b))
# 		# log_prob_sum = tf.math.reduce_sum(log_probs)
# 		# #objectives.append(log_prob_sum)
		
# 		objective = -log_prob_sum
# 		grads = actor_tape.gradient(objective, actor_net.trainable_variables)
# 		actor_optimizer.apply_gradients( zip(grads, actor_net.trainable_variables) )

		#TODO: compute the final objective to optimize, is the average between all the considered trajectories
		#raise NotImplementedError
	