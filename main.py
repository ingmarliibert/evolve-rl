import argparse
import sys
import torch

import gym
from gym import wrappers, logger
from agent import *
import numpy as np
import copy
import time

def run_agents(agents):
    
    reward_agents = []
    env = gym.make("CartPole-v0")
    
    for agent in agents:
        agent.eval()

        observation = env.reset()
        
        r=0
        s=0
        
        for _ in range(250):
            
            inp = torch.tensor(observation).type('torch.FloatTensor').view(1,-1)
            output_probabilities = agent(inp).detach().numpy()[0]
            action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
            new_observation, reward, done, info = env.step(action)
            r=r+reward
            
            s=s+1
            observation = new_observation

            if(done):
                break

        reward_agents.append(r)        
        #reward_agents.append(s)
        
    
    return reward_agents

def return_average_score(agent, runs):
    score = 0.
    for i in range(runs):
        score += run_agents([agent])[0]
    return score/runs

def run_agents_n_times(agents, runs):
    avg_score = []
    for agent in agents:
        avg_score.append(return_average_score(agent,runs))
    return avg_score

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


game_actions = 2 #2 actions possible: left or right
num_agents = 100
# How many top agents to consider as parents
top_limit = 10
# run evolution until X generations
generations = 40

if __name__ == '__main__':
    #disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize N number of agents
    agents = return_random_agents(num_agents)

    elite_index = None

    for generation in range(generations):
        # return rewards of agents
        rewards = run_agents_n_times(agents, 3) #return average of 3 runs

        # sort by rewards
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] #reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
        print("")
        print("")
        
        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])
        
        print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
        #print(rewards)
        print("Top ",top_limit," scores", sorted_parent_indexes)
        print("Rewards for top: ",top_rewards)
        
        # setup an empty list for containing children agents
        children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index)

        # kill all agents, and replace them with their children
        agents = children_agents
    
    print("Testing")
    agent = agents[elite_index]
    agent.eval()
    outdir = '/tmp/random-agent-results'
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env, directory=outdir, force=True)
    ob = env.reset()
    env.render()
    reward = 0
    done = False
    while True: 
        inp = torch.tensor(ob).type('torch.FloatTensor').view(1,-1)
        output_probabilities = agent(inp).detach().numpy()[0]
        action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
        ob, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.1)
        if done:
            break