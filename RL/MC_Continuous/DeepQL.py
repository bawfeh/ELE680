import numpy as np
from matplotlib import pyplot as plt
import cv2
# import copy
from os.path import join
import torchvision.transforms as T
# from PIL import Image

from MC_Continuous.CNN import torch, CNN

class Agent():
    def __init__(self, 
                env, 
                epsilon=0.1, 
                max_epsilon=1, 
                gamma=0.9, 
                path=None, 
                recording_range=None, 
                num_actions=10,
                epsilon_decay=200,
                seed=42):  
        """
        Class Inputs:
                env : an instance of the gym environmnent
            epsilon : minimum exploration paramater for greedy action selection | float (between 0 and 1)
        max_epsilon : maximum exploration paramater for greedy action selection | float (between 0 and 1)
              gamma : discount factor for computing expected discounted rewards | float (between 0 and 1)
               path : directory where to save trained model | string, default=None
               seed : random generator seed for results reproducibility | int (positive)
    recording_range : range of episodes for video recording | list<int> (nonnegative)
        num_actions : number of discrete actions | int (positive)
        """ 
        # Torchvision tool for image transformations (includes min-Max scaling)
        # Frames are resized to take only small windows with car at center
        self.resize = T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.Resize(40, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor()
        ]) 

        # if GPU is to be used
        self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else
                    "mps" if torch.backends.mps.is_available() else
                    "cpu"
                )

        np.random.seed(seed)
        torch.seed = seed
        torch.manual_seed(seed)

        # Set the name of the environment and reset  
        self.env = env
        self.env.reset(seed=seed)
        self.recording = hasattr(env, 'recording')  # environment prepared to recode episodes
        self.num_states = env.observation_space.shape[0]
        self.num_actions = num_actions # env.action_space.shape[0]

        # Discretize the action space
        self.discrete_actions_limits = \
            np.linspace(env.action_space.low.item(), 
                        env.action_space.high.item(), num_actions+1) # discretized action space
        self.discrete_actions = np.mean((self.discrete_actions_limits[:-1], 
                                         self.discrete_actions_limits[1:]), axis=0) # midpoints

        # Check recording range:
        self.recording_range = recording_range
        self.recording = self.recording and (recording_range is not None)

        # Specify path for saving models and other infos
        self.path = path
           
        # Set up the epsilon value
        self.minEpsilon = min(max(0, epsilon), 1)
        self.epsilon = min(max(0, max_epsilon), 1)
        self.maxEpsilon = max_epsilon
        self.EPS_DECAY = epsilon_decay

        # Set up the gamma value (decay factor)
        self.gamma = gamma  

        # list of captured video frames
        self.frames = []
        _, self.frame_width, self.frame_height = \
            self.resize(self.get_frame()).shape

        # # Initialize Neural Net framework
        self.createNeuralNetwork()

    def createNeuralNetwork(self, lr=0.001):
        """
        - Create neural network model that will serve as a agent network.
        - Define optimizers and loss functions 

        ARGS:
            lr : learning rate for training the neural network | float (between 0 and 1)
        """
        # Generate instance of the primary model
        self.model = CNN(h=self.frame_height, w=self.frame_width, outputs=self.num_actions)

        # Set up the optimizers for both primary and secondary the models
        self.modelOptimizer = torch.optim.Adam(self.model.parameters(), lr=lr)       

        # Set the loss function
        self.lossfun = torch.nn.MSELoss() 

    def actionChoice(self, Q, train_mode=True):
        """ Use an epsilon-greedy strategy for action selection (given Q values) """
        if np.random.random() < self.epsilon and train_mode:
            # action  = self.env.action_space.sample() # Randomly choose an action
            action = np.random.choice(self.num_actions) # Randomly choose an action
        else:
            action = np.argmax(Q) # Choose the action predicted by model
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
        Q_ = Q.cpu().data.numpy()
        # Select an action using predicted Q values
        action = self.actionChoice(Q_, train_mode)
        action_ = np.array([self.discrete_actions[action]])
        
        # Step through the environment using action_
        next_, reward, done, _, _ = self.env.step(action_)
        next_state = torch.from_numpy(next_).float() 

        if render:
            self.capture_video_frame(episode)

        return (next_state, action, reward, done)
    
    def update_epsilon(self, steps):
        """ updates the value of epsilon within each episode """
        self.epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * \
        np.exp(-1. * steps / self.EPS_DECAY)

    

    # Run tests on the trained model
    def test(self, numTest=1, maxMoves = None, render=True, saved_model_path=None): 
        """ Testing the Neural model """       
        # All all the rewards for the number of games played
        allRewards = []
        self.frames = []
        if maxMoves is None:
            maxMoves = self.env.spec.max_episode_steps

        if saved_model_path is not None:
            model = torch.load(saved_model_path, weights_only=False)
            self.model.load_state_dict(model)

        info = {}  # traces information across the steps of the episode   
        wins = 0   # counts number of games won
        for g in range(numTest):
            info[g] = dict(actions=[], rewards=[], terminated=False)
            # Initialize the environment and state                                                         
            _ = self.env.reset()
            totalReward = 0
            frame = self.resize(self.get_frame()).unsqueeze(0) #.to(device)
            next_frame = self.resize(self.get_frame()).unsqueeze(0) #.to(device)
            # Difference in rendered frames take as input observed state for NN
            state = next_frame - frame
            done = False # Is game over?    
            moves = 0
            # Play while game not over or max number of moves hasnt been played
            while (not done) and (moves < maxMoves):
                moves += 1
                if render:
                    self.capture_video_frame(all=True)
                Q = self.model(state) # Pass the state through the trained model
                Q_ = Q.detach().numpy() 
                action = np.argmax(Q_) 
                action_ = np.array([self.discrete_actions[action]])                             
                _, reward, done, _, _ = self.env.step(action_)
                frame = next_frame
                next_frame = self.resize(self.get_frame()).unsqueeze(0) #.to(device)
                # Cumulatively store the received rewards
                totalReward += reward
                # Set the next state as current state
                state = next_frame - frame
                if render:
                    self.capture_video_frame(all=True)
                # trace information
                info[g]['actions'].append(action_.item())
                info[g]['rewards'].append(reward)
                info[g]['terminated'] = done
            
            print('Test {}: total rewards = {:.2f} (in {}/{} moves)' \
                  .format(g, totalReward, moves, maxMoves))
            allRewards.append(totalReward)
            if done or (moves < maxMoves):
                wins += 1
        # Create video (if any)
        self.create_video(mode='test')

        print('='*15+"\n  Summary\n"+'-'*15 \
              +"\nplays: {} \nwins: {}\nmean (rewards): {} \nstd (rewards): {}" \
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

    def save_model(self, history, metric='reward', tol=0.01, m_averages=100, filename=None):
        """ Save best model based on moving averages of specified metric """
        if (len(history[metric])>m_averages) and (self.path is not None):
            fname = 'saved_model' if (filename is None) else filename
            last = -m_averages
            if (np.mean(history[metric][last:]) \
                - np.mean(history[metric][last-1:-1])) > tol:
                torch.save(self.model.state_dict(), self.path+fname+'.pt')
                return True
        return False

    
    # Courtesy: https://github.com/greatwallet/mountain-car?tab=readme-ov-file
    def get_car_location(self, frame_width):
        xmin = self.env.env.min_position
        xmax = self.env.env.max_position
        world_width = xmax - xmin
        scale = frame_width / world_width
        return int(self.env.state[0] * scale + frame_width / 2.0)  # MIDDLE OF CAR

    # Courtesy: https://github.com/greatwallet/mountain-car?tab=readme-ov-file
    def get_frame(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        frame = self.env.render() #(mode='rgb_array')
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, frame_width, _ = frame.shape
        # screen = screen[int(screen_height * 0.8), :]
        view_width = int(frame_width)
        car_location = self.get_car_location(frame_width)
        if car_location < view_width // 2:
            slice_range = slice(frame_width)
        elif car_location > (frame_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(car_location - view_width // 2,
                                car_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        frame = frame[:, slice_range, :]
        return  frame
