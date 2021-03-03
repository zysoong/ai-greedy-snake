# AI Snake
Experimental project: Snake game AI using reinforcement learning\
The agent runs purely with the console, which makes it easy to be trained with a cloud server.\
The following animated image shows a training result via the DDQN model. \
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
batch size = 32, gamma = 0.97, **critic lr = 0.01, non-decay**\
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/ddqn_plot.png?raw=true)\
batch size = 32, gamma = 0.97, **critic lr = 0.01, critic lr decay = 0.999995**\
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/ddqn_plot_d0999995.png?raw=true)\
batch size = 32, gamma = 0.97, **critic lr = 0.0025, non-decay**\
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/ddqn_plot_.png?raw=true)\
batch size = 32, gamma = 0.97, **critic lr = 0.0025, critic lr decay = 0.999995**\
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/ddqn_plot_d0999995_.png?raw=true)\

## ADHDP with 8-dimensional state
(working)\
Blue line: batch size = 32, gamma = 0.97, critic lr = 0.0025, actor lr = 0.0025, non-decay
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/adhdp_plot.png?raw=true)\
The training history shows that the ADHDP agent ends up overfitting if the decay of the learning rate is not applied, especially on the critic-net. For further planning, decaying of learning rates, applying learning rates according to the scores, or experience replay will be tried to see if better results would come out. \
Early stopping is currently recommended to avoid the problem of overfitting.

# Conclusion
(working)
