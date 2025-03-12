
# Reinforcement Learning: Q-Learning in FrozenLake Environment

## About This Project
This repository contains a project focused on Reinforcement Learning (RL), specifically using the Q-Learning algorithm. It is part of university coursework at Vilnius Tech. In this project, we explore how to train an AI agent to navigate the FrozenLake environment using Q-Learning, an approach in which the agent learns optimal actions through interaction with the environment.

## Purpose
The goal of this project is to demonstrate how an agent can learn to navigate through an environment, maximizing its rewards over time by exploring, taking actions, and learning from the results. The FrozenLake environment is used to illustrate the concept of Q-Learning in action, where the agent's objective is to find the goal without falling through holes in a frozen lake.

## Features
- Implementation of the Q-Learning algorithm for training an agent
- Exploration of the FrozenLake environment using OpenAI Gym
- Step-by-step updates to the Q-Table based on the agent's experience
- Visualization of the training progress through rewards over episodes
- Implementation of an epsilon-greedy approach for balancing exploration and exploitation

## How to Use
1. Install the required dependencies:
    ```bash
    pip install gym numpy matplotlib
    ```

2. Import the necessary libraries and set up the environment:
    ```python
    import gym
    import numpy as np
    import matplotlib.pyplot as plt
    ```

3. Define constants, set up the Q-Table, and begin training:
    ```python
    env = gym.make('FrozenLake-v0')
    STATES = env.observation_space.n
    ACTIONS = env.action_space.n
    Q = np.zeros((STATES, ACTIONS))

    # Define learning parameters
    EPISODES = 1500
    MAX_STEPS = 100
    LEARNING_RATE = 0.81
    GAMMA = 0.96
    epsilon = 0.9

    # Train the agent
    rewards = []
    for episode in range(EPISODES):
        state = env.reset()
        for _ in range(MAX_STEPS):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, done, _ = env.step(action)
            Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state
            if done:
                rewards.append(reward)
                epsilon -= 0.001
                break
    ```

4. After training, visualize the agent's performance:
    ```python
    # Plot training progress
    avg_rewards = [sum(rewards[i:i+100])/100 for i in range(0, len(rewards), 100)]
    plt.plot(avg_rewards)
    plt.ylabel('Average Reward')
    plt.xlabel('Episodes (100\'s)')
    plt.show()
    ```

## Sources
- Violante, Andre. “Simple Reinforcement Learning: Q-Learning.” Medium, Towards Data Science, 1 July 2019, https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56.
- OpenAI. “OpenAI/Gym.” GitHub, https://github.com/openai/gym/wiki/FrozenLake-v0.
