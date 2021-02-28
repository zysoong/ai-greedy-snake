# AI Snake
Experimental project: Snake game AI using reinforcement learning\
The agent runs purely with the console, which makes it easy to be trained with a cloud server
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/example_ddqn_reduced.gif?raw=true)

# Getting started
(working)

# Principles
(working)

# Best results
The best results will be updated consistently, as far as a better result is available
Model | Highest avg. score of 1000 steps | Comment
--- | --- | ---
DQN with 8-dimensional state | 17.005 | -
DDQN with 8-dimensional state | 25.093 | - 
Actor-Critic (ADHDP) with 8-dimensional state | 15.351 | -
DDQN with CNN | 4.0  | infinite rotation, Hit rate = 0.27%
Actor-Critic (ADHDP) with CNN | 3.0  | infinite rotation, Hit rate = 0.01%
Actor-Critic-Target | (working) | (working)

# Training histories
## DQN with 8-dimensional state
(working)
## DDQN with 8-dimensional state
Blue line: batch size = 32, gamma = 0.97, critic lr = 0.01, non-decay
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/ddqn_plot.png?raw=true)
The training history shows that the DDQN agent ends up overfitting if the decay of the learning rate is not applied. For further planning, I will try with decay and different learning rates, applying learning rates according to the scores, or experience replay to see if we could get better results. \
(working)
## ADHDP with 8-dimensional state
Blue line: batch size = 32, gamma = 0.97, critic lr = 0.0025, actor lr = 0.0025, non-decay
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/adhdp_plot.png?raw=true)
The training history shows that the ADHDP agent ends up overfitting if the decay of the learning rate is not applied. For further planning, I will try with decay and different learning rates, applying learning rates according to the scores, or experience replay to see if we could get better results. \
(working)
