import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from main import *

class CartPoleAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
                    nn.Linear(4,128, bias=True),
                    nn.ReLU(),
                    nn.Linear(128,2, bias=True),
                    nn.Softmax(dim=1)
                    )

            
    def forward(self, inputs):
        x = self.fc(inputs)
        return x

def init_weights(m):
    # nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
    # nn.Conv2d bias is of shape [16] i.e. # number of filters
    
    # nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
    # nn.Linear bias is of shape [32] i.e. # number of output features
    
    if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)

def return_random_agents(num_agents):
    
    agents = []
    for _ in range(num_agents):
        
        agent = CartPoleAI()
        
        for param in agent.parameters():
            param.requires_grad = False
            
        init_weights(agent)
        agents.append(agent)
        
        
    return agents


def mutate(agent):

    child_agent = copy.deepcopy(agent)
    
    mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
            

    for param in child_agent.parameters():

        mutation = np.random.randn(*tuple(param.shape))
            
        param += torch.from_numpy(mutation_power*mutation) 
        
    return child_agent
    
    
def apply_crossover(parent1, parent2):
    new_agent = copy.deepcopy(parent1)
    rng = np.random.default_rng()
    for i, param in enumerate(new_agent.parameters()):
        mask = rng.integers(0, 2, size=(tuple(param.shape)))
        
        param = np.where(mask == 0, list(parent1.parameters())[i], list(parent2.parameters())[i])
        
    return new_agent
        
        
    
def crossover(good_parents):
    
    crossovers = []
    
    for i in range(len(good_parents)):
        crossovers.append(good_parents[i])
        
        idxs = np.random.choice(len(good_parents), replace=False, size=2)
        
        parent1 = good_parents[idxs[0]]
        parent2 = good_parents[idxs[1]]
        
        crossovers.append(apply_crossover(parent1, parent2))
        
    return crossovers
    

def return_children(agents, sorted_parent_indexes, elite_index, cross_over=True):
    
    children_agents = []
    
    good_parents = [agents[i] for i in sorted_parent_indexes]
        
    if cross_over:
        
        good_parents = crossover(good_parents)
    
    #first take selected parents from sorted_parent_indexes and generate N-1 children
    for i in range(len(agents)-1):
        
        selected_agent_index = np.random.randint(len(good_parents))
        children_agents.append(mutate(good_parents[selected_agent_index]))

    #now add one elite
    elite_child, top_elite_score = add_elite(agents, sorted_parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index=len(children_agents)-1 #it is the last one
    
    return children_agents, elite_index, top_elite_score

def add_elite(agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):
    
    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]
    
    if(elite_index is not None):
        candidate_elite_index = np.append(candidate_elite_index,[elite_index])
        
    top_score = None
    top_elite_index = None
    
    for i in candidate_elite_index:
        score = return_average_score(agents[i],runs=5)
        print("Score for elite i ", i, " is ", score)
        
        if(top_score is None):
            top_score = score
            top_elite_index = i
        elif(score > top_score):
            top_score = score
            top_elite_index = i
            
    print("Elite selected with index ",top_elite_index, " and score", top_score)
    
    child_agent = copy.deepcopy(agents[top_elite_index])
    return child_agent, top_score
