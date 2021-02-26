# AI Snake
Experimental project: Snake game AI using reinforcement learning\
The agent runs purely with the console, which makes it easy to be trained with a cloud server
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/example_ddqn_reduced.gif?raw=true)

# Getting started
(working)

# Best results
Model | Coding completed | Highest avg. score of 1000 steps | Comment
--- | --- | --- | ---
DQN with 8-dimensional state | Yes | 17.005 | -
DDQN with 8-dimensional state | Yes | 25.093 | - 
Actor-Critic (ADHDP) with 8-dimensional state | Yes | 15.351 | -
DDQN with CNN | Yes | 4.0  | infinite rotation, Hit rate = 0.27%
Actor-Critic (ADHDP) with CNN | Yes | 3.0  | infinite rotation, Hit rate = 0.01%
Actor-Critic-Target | No | - | -

# Training histories
## DQN with 8-dimensional state
(working)
## DDQN with 8-dimensional state
(working)
## ADHDP with 8-dimensional state
Blue line: batch size = 32, gamma = 0.97, critic lr = 0.0025, actor lr = 0.0025, non decay
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/adhdp_plot.png?raw=true)
