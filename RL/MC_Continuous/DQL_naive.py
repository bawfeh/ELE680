from MountainCar.DeepQL import *

class naive_Agent(Agent):
    def __init__(self, env, **kwargs): 
        super(naive_Agent, self).__init__(env=env, **kwargs)

    def learn_online(self, Q, next_state, reward, done):
        # State action values (associated with the actions that were actually taken)
        Y = Q.squeeze()
        # Calculate the Q values for the next state
        with torch.no_grad():
            QNext = self.model(next_state)
        # Next state action values
        YHat = reward if done else (reward + self.gamma * torch.max(QNext, dim=1)[0])
        
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
              render=False): 
        """ Training the Neural model """
        # Track losses and cumulative expected rewards 
        history = {'loss':[], 'reward':[], 'wins':[]}   
        # Initialize Neural Net framework
        self.createNeuralNetwork(lr=learningRate)
        
         # Don't render while recording
        # render = render and not(self.recording)
        self.frames = []
        if maxMoves is None:
            maxMoves = self.env.spec.max_episode_steps

        for e in range(epochs):                                                            
            # Store the initial state
            state = torch.from_numpy(self.env.reset()[0]).float()
            # Store the rewards for each game played
            totalReward = 0
            # Continue until game over (done=True) 
            # or the specified number of max moves are reached
            done = False
            moves = 0
            while (not done) and (moves < maxMoves):
                moves += 1
                next_state, Q, reward, done = self.step(state, render, e)
                loss = self.learn_online(Q, next_state, reward, done)
                # Store the cumulative reward
                totalReward += reward
                # Assign the next state as current
                state = next_state

            # Store the cumulative reward and wins
            history['reward'].append(totalReward)
            history['wins'].append(done)
            history['loss'].append(loss.item())

            # Adapt the epsilon value in each epoch
            if self.epsilon > self.minEpsilon:
                self.epsilon -= self.maxEpsilon / epochs   

            # Print the progress 
            if e % np.round(epochs/10) == 0:                 
                error = np.mean(history['loss'])
                print('epoch {:d}: \terror = {:.2e}, cumulative reward = {:.2f} (in {} moves)' \
                      .format(e, error, totalReward, self.env._elapsed_steps))

        self.env.close()

        # Create video (if any)
        self.create_video()

        print(f"Wins {sum(history['wins'])} out of {epochs} plays!")
        
        # Return training history
        return history