from MC_Continuous.DeepQL import *
from collections import deque, namedtuple

class experience_Agent(Agent):
    def __init__(self, env, **kwargs): 
        super(experience_Agent, self).__init__(env=env, **kwargs)

    def learn_from_experience(self, replay, batchSize):
        """ Uses batch of info from experiences gathered in a memory buffer (replay)
         to train agent network. """
        # Retrieve a random batch of information (state, action, reward, done, next) from replay memory                             
        batchIndices = np.random.choice(len(replay), size=batchSize, replace=False)  
        stateBatch = torch.stack([replay[i].state for i in batchIndices]).squeeze(1) 
        actionBatch = torch.Tensor([replay[i].action for i in batchIndices]) 
        rewardBatch = torch.Tensor([replay[i].reward for i in batchIndices]) 
        doneBatch = torch.Tensor([replay[i].done for i in batchIndices]) 
        nextStateBatch = torch.stack([replay[i].next_state for i in batchIndices]).squeeze(1)  
        
        # Calculate the Q values for the current state
        Q = self.model(stateBatch)
        # State action values (associated with the actions that were actually taken)
        Y = Q.gather(dim=1, index=actionBatch.long().unsqueeze(dim=1)).squeeze()

        # Calculate the Q values for the next state
        with torch.no_grad():
            QNext = self.model(nextStateBatch)
        # One-step returns (Gt:t+1) using discounted Q values expected on next state [terminal value = 0]
        YHat = rewardBatch + self.gamma * ((1 - doneBatch) * torch.max(QNext, dim=1)[0])
        
        # Back propagate
        loss = self.lossfun(Y, YHat)
        self.modelOptimizer.zero_grad()
        loss.backward()
        self.modelOptimizer.step()
        return loss

    def train(self, 
              epochs=1000, 
              learningRate=0.001, 
              maxMoves = None, 
              memorySize=250, 
              batchSize=100, 
              render=False): 
        """ Training the Neural model """
        # Ensure memorySize >= batchSize
        assert  memorySize >= batchSize
        # Track losses and cumulative expected rewards 
        history = {'loss':[], 'reward':[], 'wins': []}   
        # Set up the memory for experiance replay
        replay = deque(maxlen=memorySize)
        Experience = namedtuple('buffer', 
            field_names=['state', 'action', 'reward', 'done', 'next_state']
        )
        # Initialize Neural Net framework
        self.createNeuralNetwork(lr=learningRate)
        
        self.frames = []  # to collect frames for video
        if maxMoves is None:
            maxMoves = self.env.spec.max_episode_steps
        win_size = epochs // 10 # window size for moving averages
        best_episode = 0

        for e in range(epochs):                                                            
            # Store the initial state
            frame = self.resize(self.get_frame()).unsqueeze(0) #.to(device)
            next_frame = self.resize(self.get_frame()).unsqueeze(0) #.to(device)
            state = next_frame - frame
            # state = torch.from_numpy(self.env.reset()[0]).float()
            # Store the rewards for each game played
            totalReward = 0
            # Continue until game over (done=True) 
            # or the specified number of max moves are reached
            done = False
            moves = 0
            while (not done) and (moves < maxMoves):
                self.update_epsilon(moves)
                moves += 1
                _, action, reward, done = self.step(state, render, e)
                frame = next_frame
                next_frame = self.resize(self.get_frame()).unsqueeze(0) #.to(device)
                next_state = next_frame - frame
                # Store the cumulative reward
                totalReward += reward
                # Add to memory
                replay.append(Experience(state, action, reward, done, next_state))
                # Assign the next state as current
                state = next_state

            # Store the cumulative reward and wins
            history['reward'].append(totalReward)
            history['wins'].append(done)

            if len(replay) > batchSize: 
                loss = self.learn_from_experience(replay, batchSize)
                history['loss'].append(loss.item())

            # # Adapt the epsilon value in each epoch
            # if self.epsilon > self.minEpsilon:
            #     self.epsilon -= self.maxEpsilon / epochs  

            # Save best model
            saved = self.save_model(history) 
            if saved: best_episode = e

            # Print the progress 
            if e % win_size == 0:  
                error = np.mean(history['loss'][-win_size:]) if len(history['loss']) else np.nan
                if np.isnan(error):
                    print('epoch {:d}: \terror = {}, cumulative reward = {:.2f} (in {} moves)' \
                        .format(e, error, totalReward, moves))
                else:
                    print('epoch {:d}: \terror = {:.2e}, cumulative reward = {:.2f} (in {} moves)' \
                        .format(e, error, totalReward, moves))

        self.env.close()

        # Create video (if any)
        self.create_video()

        print(f"Wins {sum(history['wins'])} out of {epochs} plays!")
        print(f"Saved episode {best_episode}!")
        
        # Return training history
        return history