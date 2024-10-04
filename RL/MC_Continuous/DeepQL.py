import numpy as np
from matplotlib import pyplot as plt
import cv2
# import copy
from os.path import join

from MC_Continuous.ANN import torch, ANN


class Agent():
    def __init__(self, env, layers=None, epsilon=0.1, max_epsilon=1, gamma=0.9, algorithm=None, recording_range=None, seed=42):  
        """
        Class Inputs:
            - env : an instance of the gym environmnent
            - layers : positive integers representing the number of units per ANN hidden layer | list<int> , default=None
            - epsilon : minimum exploration paramater for greedy action selection | float (between 0 and 1)
            - gamma : discount factor for computing expected discounted rewards | float (between 0 and 1)
            - seed : random generator seed for results reproducibility | int (positive)
            - recording_range : range of episodes for video recording | list<int> (nonnegative)
            - algorithm : flag that determines agent training algorithm used  | string type | 
                'naive' [online value estimator or Vanilla QL] (default), 
                'replay' [QL with experience replay], 
                'target' [QL with target network]
        """ 
        # Set the name of the environment and reset  
        self.env = env
        self.env.reset(seed=seed)
        self.recording = hasattr(env, 'recording')  # environment prepared to recode episodes
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]

        # Check recording range:
        self.recording_range = recording_range
        self.recording = self.recording and (recording_range is not None)

        # Choose an algorithm
        self.algorithm = algorithm

        # Set up ANN hidden layers configuration
        self.layers  = [] if (layers is None) else layers
           
        # Set up the epsilon value
        self.minEpsilon = min(max(0, epsilon), 1)
        self.epsilon = min(max(0, max_epsilon), 1)
        self.maxEpsilon = max_epsilon

        # Set up the gamma value (decay factor)
        self.gamma = gamma  

        # Set list for capturing video frames
        self.frames = []

        # Initialize Neural Net framework
        self.createNeuralNetwork()

    def createNeuralNetwork(self, lr=0.001):
        """
        - Create neural network model that will serve as a agent network.
        - Define optimizers and loss functions 

        ARGS:
            lr : learning rate for training the neural network | float (between 0 and 1)
        """
        # Dynamically generate the primary model
        self.model = ANN(input_dim=self.num_states, output_dim=self.num_actions,
                          layers=self.layers, activation='tanh')

        # Set up the optimizers for both the models
        self.modelOptimizer = torch.optim.Adam(self.model.parameters(), lr=lr)       

        # Set the loss function
        self.lossfun = torch.nn.MSELoss()   

    def actionChoice(self, Q, train_mode=True):
        """ Use an epsilon-greedy strategy for action selection (given Q values) """
        if np.random.random() < self.epsilon and train_mode:
            action  = self.env.action_space.sample() # Randomly choose an action
        else:
            action = Q # Choose the action predicted by model
        return action 
    
    def capture_video_frame(self, episode=0, all=False):
        if self.recording:
            if (episode in self.recording_range) or all:
                self.frames.append(self.env.render())
    
    def create_video(self, mode=None):
        mode_ = 'train' if (mode is None) else 'test'
        mode_ = '_' + mode_
        if len(self.frames):
            frame_size = self.frames[0].shape[::-1][1:]
            filename = join(self.env.video_folder, self.env.name_prefix + mode_ + '.mp4')
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 60, frame_size)
            for frame in self.frames:
                out.write(frame)

    def step(self, state, render=False, episode=0, train_mode=True):
        """ one step of the game given current state """
        if render:
            self.capture_video_frame(episode)
        # Forward pass on current state (returns Q values for each action)
        Q = self.model(state)
        Q_ = Q.data.numpy()
        # Select an action using predicted Q values
        action = self.actionChoice(Q_, train_mode)
        
        # Step through the environment using action
        next_, reward, done, _, _ = self.env.step(action)
        next_state = torch.from_numpy(next_).float() 
        return (next_state, action, reward, done)

    
    # Run tests on the trained model
    def test(self, numTest=1, maxMoves = None, render=True): 
        """ Testing the Neural model """       
        # All all the rewards for the number of games played
        allRewards = []
        self.frames = []
        if maxMoves is None:
            maxMoves = self.env.spec.max_episode_steps

        info = {}  # traces information across the steps of the episode
        for g in range(numTest):
            info[g] = dict(actions=[], rewards=[], terminated=False)
            # Initial state
            totalReward = 0
            state = torch.from_numpy(self.env.reset()[0]).float()
            done = False # Is game over?       
            wins = 0
            moves = 0
            # Play while game not over or max number of moves hasnt been played
            while (not done) and (moves < maxMoves):
                moves += 1
                state, action, reward, done = self.step(state, render, train_mode=False)
                # trace information
                info[g]['actions'].append(action.item())
                info[g]['rewards'].append(reward)
                info[g]['terminated'] = done
            
            print('Test episode {}: total rewards = {:.2f} (in {}/{} moves)' \
                  .format(g, totalReward, moves, maxMoves))
            allRewards.append(totalReward)
            if done:
                wins += 1
        # Create video (if any)
        self.create_video(mode='test')

        print('='*15+"\n  Summary\n"+'-'*15+"\nplays: {} \nwins: {} \nstd: {}" \
              .format(numTest, wins, np.mean(allRewards), np.std(allRewards)))

        # Return information on each episode
        return info 


    def moving_average(self, v, window_size=10):
        m_average = np.mean([v[l:l+window_size] \
                 for l in range(len(v)-window_size+1)], axis=1) \
                      if window_size>0 else v
        return m_average
    
    def axis_format(self, ax, on, metric):
        ax.grid(on)
        ax.set(xlabel='episode', ylabel=metric)

    def plot_history(self, history, grid_on=True, m_averages=0, metric=None):
        if metric is None:
            _, axs = plt.subplots(1, 2, figsize=(14,6))
            axs[0].plot(self.moving_average(history['loss'], m_averages))
            self.axis_format(axs[0], grid_on, 'loss')
            axs[1].plot(self.moving_average(history['reward'], m_averages))
            self.axis_format(axs[1], grid_on, 'reward')
            plt.show()  
        else:  # make on plot only
            fig = plt.figure(figsize=(8,7))
            ax = fig.add_subplot(1,1,1)
            ax.plot(self.moving_average(history[metric], m_averages))
            self.axis_format(ax, grid_on, metric)
            plt.show()
