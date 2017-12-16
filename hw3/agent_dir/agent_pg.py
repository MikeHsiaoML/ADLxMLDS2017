import math
import random
from collections import deque 
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from agent_dir.agent import Agent
import scipy
import scipy.misc
import numpy as np

FloatTensor = torch.cuda.FloatTensor 
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor
Tensor = FloatTensor

def prepro(o, image_size=[80, 80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py

    Input:
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array
        Grayscale image, shape: (80, 80, 1)

    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32), axis=2)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.linear1 = nn.Linear(32*8*8, 128)
        self.linear2 = nn.Linear(128,3)
        self.softmax = nn.Softmax()
        self.reward_list = []
        self.log_prob_list = []
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.linear1(x.view(x.size(0),-1)))
        return self.softmax(self.linear2(x))

class Agent_PG(Agent):
    ''' PG Agent '''

    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG, self).__init__(env)
        self.gamma = 0.99
        self.model = Policy()
        self.opt = optim.RMSprop(self.model.parameters(),lr = 1e-4, weight_decay = 0.99)
        self.save_every = 100000
        self.reward_queue = deque([])
        self.model.train()
        self.prev_state = np.zeros((80, 80, 1))

        if args.test_pg:
            # you can load your model here
            checkpoint = torch.load('31300000.tar')
            self.model = Policy()
            self.model.load_state_dict(checkpoint['policy'])
            self.model.eval()

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        self.model.eval()
        self.prev_state = np.zeros((80, 80, 1))

    def train(self):
        """
        Implement your training algorithm here
        """
        time_step = 0
        done = True
        for i_episode in count():
            epi_reward = 0
            if done:
                raw_current_state, done = self.env.reset(), False
                self.prev_state = np.zeros((80, 80, 1))

            while not done:
                time_step += 1
                action = self.make_action(raw_current_state,False)
                raw_current_state, reward, done, _ = self.env.step(action)
                self.model.reward_list.append(reward)
                epi_reward += reward
                
                if (time_step+1)%self.save_every == 0:
                    torch.save({'policy':self.model.state_dict(),'opt':self.opt.state_dict(),'reward_queue':self.reward_queue},
                                'pg_{}.tar'.format(time_step+1)) 
             
                if done:
                    self.reward_queue.append(epi_reward)
                    if len(self.reward_queue) > 30:
                        self.reward_queue.popleft()
                    print("Episode {}, time step {}, Reward = {}".format(i_episode+1, time_step+1, epi_reward))
                    self.optimize_model()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        current_state = prepro(observation)
        diff_state = current_state - self.prev_state
        self.prev_state = current_state
        
        input_state = Variable(torch.from_numpy(diff_state).float().view(1, 1, 80, 80))

        probs = self.model(input_state)
        action = probs.multinomial().data[0, 0] 
        if not test:
            self.model.log_prob_list.append(probs[:, action].log())
        action += 1

        return action

    def optimize_model(self):
        ''' Update PG Network '''
        policy_loss = []
        running_reward = 0
        running_reward_list = []
        for r in reversed(self.model.reward_list):
            if r != 0:  # when one player gets points
                running_reward = 0
            running_reward = r + self.gamma * running_reward
            running_reward_list.insert(0, running_reward)

        running_reward_list = Tensor(running_reward_list)
        running_reward_list = (running_reward_list - running_reward_list.mean()) / \
            (running_reward_list.std() + np.finfo(np.float32).eps)

        for log_prob, reward in zip(self.model.log_prob_list, running_reward_list):
            policy_loss.append(-log_prob * reward)

        self.opt.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.opt.step()

        del self.model.log_prob_list[:]
        del self.model.reward_list[:]
