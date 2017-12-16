import math
import random
import numpy as np
from collections import namedtuple,deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from agent_dir.agent import Agent

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

FloatTensor = torch.cuda.FloatTensor 
LongTensor = torch.cuda.LongTensor
ByteTensor = torch.cuda.ByteTensor
Tensor = FloatTensor

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.linear1 = nn.Linear(64*7*7, 512)
        self.linear2 = nn.Linear(512,4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.linear2(F.relu(self.linear1(x.view(x.size(0),-1))))
    
class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 0.95
        self.eps_end = 0.05
        self.eps_decay = 500000 
        self.Q = DQN().cuda()
        self.target_Q = DQN().cuda()
        self.opt = optim.RMSprop(self.Q.parameters(),lr = 0.00025)
        self.memory = ReplayMemory(100000)
        self.reward_queue = deque([])
        self.steps_done = 0
        self.save_every = 100000
        self.learning_start = 10000
        self.learning_freq = 4
        self.param_updates = 0
        self.target_update_freq = 100

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        checkpoint = torch.load('8800000.tar')
        self.Q = DQN().cuda()
        self.Q.load_state_dict(checkpoint['Q'])
        self.target_Q = DQN().cuda()
        self.target_Q.load_state_dict(checkpoint['target_Q'])
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in batch.next_state if s is not None]), volatile=True).cuda()
        state_batch = Variable(torch.cat(batch.state)).cuda()
        action_batch = Variable(torch.cat(batch.action)).cuda()
        reward_batch = Variable(torch.cat(batch.reward)).cuda()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.Q(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        next_state_values = Variable(torch.zeros(self.batch_size).type(Tensor))
        next_state_values[non_final_mask] = self.target_Q(non_final_next_states).detach().max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        
        next_state_values.volatile = False
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.opt.zero_grad()
        loss.backward()
        for param in self.Q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.opt.step()
        self.param_updates += 1
        if self.param_updates % self.target_update_freq == 0:
            self.target_Q.load_state_dict(self.Q.state_dict())

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        # Initialize the environment and state
        state = self.env.reset()
        t_r = 0
        for t in count():
            if t > 10000000:
                break
            # Select and perform an action
            if t> self.learning_start:
                action = self.make_action(state,False)
            else:
                action = random.randrange(4)
            next_state, reward, done, _ = self.env.step(action)
            #print("Step {}, reward {}".format(t,reward))
            t_r += reward
            reward = Tensor([reward])
            if not done:
               # Store the transition in memory
                self.memory.push(torch.from_numpy(state).permute(2,0,1).unsqueeze(0), LongTensor([[action]]),\
                             torch.from_numpy(next_state).permute(2,0,1).unsqueeze(0), reward)
            else:
                self.memory.push(torch.from_numpy(state).permute(2,0,1).unsqueeze(0), LongTensor([[action]]),\
                            None, reward)
            
            if done:
                print("Done at time {}, total reward {}".format(t,t_r))
                self.reward_queue.append(t_r)
                if len(self.reward_queue) > 30:
                    self.reward_queue.popleft()
                t_r = 0
                next_state = self.env.reset()
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if t > self.learning_start and t % self.learning_freq == 0:
                self.optimize_model()

            if (t+1)%self.save_every == 0:
                torch.save({'Q':self.Q.state_dict(),'target_Q':self.target_Q.state_dict(),'opt':self.opt.state_dict(),'reward_queue':self.reward_queue}\
                             ,'dqn_{}.tar'.format(t+1)) 
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        state = torch.from_numpy(observation).permute(2,0,1).unsqueeze(0).cuda()
        sample = random.random()
        eps_threshold = 0.005
        if test == False:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end)*math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
        if sample > eps_threshold:
            choose_action =  self.Q(Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].cpu()
            return choose_action[0]
        else:
            random_action =  LongTensor([[random.randrange(4)]])
            return random_action[0,0]
        #return self.env.get_random_action()

