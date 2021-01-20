# Taken from https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f

import argparse
import sys
import torch

import gym
from gym import wrappers, logger
from agent import *
import numpy as np
import copy
import time
import matplotlib.pyplot as plt


def run_agents(agents):

    reward_agents = []
    env = gym.make("CartPole-v1")

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
num_agents = 500
# How many top agents to consider as parents
top_limit = 20
# run evolution until X generations
generations = 40


save_file = "./original_with_crossover.pth"

def save_agent(agent, filename):
    torch.save(agent.state_dict(), filename)

def load_agent(filename):
    agent = CartPoleAI()

    for param in agent.parameters():
        param.requires_grad = False

    agent.load_state_dict(torch.load(filename))

    return agent


def plot_graph(score_list, time_list, model_name, save_path):
    generations = np.arange(len(score_list))

    plt.plot(generations, score_list)
    plt.xlabel("generations")
    plt.ylabel("elite average score over 5 runs")
    plt.title(model_name)

    plt.savefig(save_path)



def save_data(score_list, time_list, best_model_generation, filename, cross_over, anneal, original_mutation_power, lifetime):

    score_string = ""
    for score in score_list:
        score_string += str(score)+","

    score_string = score_string[:-1]

    time_string = ""
    for time in time_list:
        time_string += str(time)+","

    time_string = time_string[:-1]

    with open(filename, "w") as f:
        f.write(score_string+"\n")
        f.write(time_string+"\n")
        f.write(str(best_model_generation)+"\n")
        f.write(f"Crossover: {cross_over}\n")
        f.write(f"anneal: {anneal}\n")
        f.write(f"original mutation power: {original_mutation_power}\n")
        f.write(f"lifetime: {lifetime}\n")

    print("SAVED!")





def train_model(model_name, max_time, cross_over, anneal, original_mutation_power, lifetime):

    print("****"*24)
    print("RUNNING MODEL", model_name)
    #disable gradients as we will not use them
    torch.set_grad_enabled(False)



    # initialize N number of agents
    agents = return_random_agents(num_agents)

    elite_index = None

    model_save_path = model_name+"_model.pth"
    data_save_path = model_name+"_data.txt"
    graph_save_path = model_name+"_graph.png"

    numpy_save_path = model_name+"_alldata"

    score_list = []
    time_list = []

    top_score = 0
    best_model_generation = -1

    generation = 0
    start_time = time.time()

    all_rewards = []

    while True:
        try:
            # return rewards of agents
            rewards = run_agents_n_times(agents, 3) #return average of 3 runs
            all_rewards.append(rewards)
            # sort by rewards
            sorted_parent_indexes = np.argsort(rewards)[-top_limit:]
            print("")
            print("")

            top_rewards = np.sort(rewards)[::-1][:top_limit]

            print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))

            # setup an empty list for containing children agents
            children_agents, elite_index, elite_score = return_children(agents, sorted_parent_indexes, elite_index, cross_over, anneal, generation, original_mutation_power, lifetime)

            print(f"Elite score: {elite_score}, previous top score: {top_score}, better than previous top: {elite_score > top_score}.")
            if elite_score > top_score:
                top_score = elite_score
                save_agent(agents[elite_index], model_save_path)
                best_model_generation = generation

            score_list.append(elite_score)
            current_time = time.time()
            time_list.append(current_time - start_time)

            if current_time-start_time > max_time:
                break

            # kill all agents, and replace them with their children
            agents = children_agents
            generation += 1
        except KeyboardInterrupt:
            #agent = agents[elite_index]
            #save_agent(agent, save_file)
            #print("SAVED!")
            break

    plot_graph(score_list, time_list, model_name, graph_save_path)
    save_data(score_list, time_list, best_model_generation, data_save_path, cross_over, anneal, original_mutation_power, lifetime)
    np.savez_compressed(numpy_save_path, data=np.array(all_rewards))

def play_agent(save_file):
    print("Testing")
    agent = load_agent(save_file)
    agent.eval()
    outdir = '/tmp/random-agent-results'
    env = gym.make("CartPole-v1")
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


if __name__ == '__main__':

    # model_name, max_time, cross_over, anneal, original_mutation_power, lifetime

    test = True

    lifetime = 40
    original_power = 0.5
    if test:
        # model_name = "test"
        # train_model(model_name, 100, True, True, original_power, lifetime)
        play_agent("crossover_annealing_model.pth")
    else:
        ONE_HOUR = 60*60

        TRAIN_TIME = 2*ONE_HOUR

        model_name = "original"
        train_model(model_name, TRAIN_TIME, False, False, original_power, lifetime)

        model_name = "crossover"
        train_model(model_name, TRAIN_TIME, True, False, original_power, lifetime)

        model_name = "annealing"
        train_model(model_name, TRAIN_TIME, False, True, original_power, lifetime)

        model_name = "crossover_annealing"
        train_model(model_name, TRAIN_TIME, True, True, original_power, lifetime)


    # play_agent(model_name+"_model.pth")

