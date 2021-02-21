# AI Snake (ongoing)
Experimental project: Snake game AI using reinforcement learning\
The agent runs purely with the console, which makes it extremely easy to be trained with a cloud server
![](https://github.com/zysoong/ai-greedy-snake/blob/master/images/example_ddqn_reduced.gif?raw=true)

# Best results
Model | Coding completed | Highest avg. score of 1000 steps | Comment
--- | --- | --- | ---
DQN with 8-dimensional state | Yes | 17.005 | -
DDQN with 8-dimensional state | Yes | 25.093 | - 
DDQN with CNN | Yes | 4.0  | infinite rotation, Hit rate = 0.27%
Actor-Critic (ADHDP) with 8-dimensional state | Yes | 15.351 
Actor-Critic (ADHDP) with CNN | Yes | 3.0  | infinite rotation, Hit rate = 0.01%
Actor-Critic-Target | No | - | -

# Training histories
(working)
