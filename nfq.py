from greedysnake import GreedySnake, Direction, Signal
import time
import numpy as np
from threading import Thread
import subprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from collections import OrderedDict
import random
import configparser
import copy
import sys
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)

class Driver:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('nfq.ini')
        self.env = config['ENV']['env']
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.max_epochs = int(config[self.env]['max_epochs'])
        self.max_steps = int(config[self.env]['max_steps'])
        self.batch_size = int(config[self.env]['batch_size'])
        self.critic_net_epochs = int(config[self.env]['critic_net_epochs'])
        self.actor_net_epochs = int(config[self.env]['actor_net_epochs'])
        self.actor_update_freq = int(config[self.env]['actor_update_freq'])
        self.gamma = float(config[self.env]['gamma'])
        self.beta_init = float(config[self.env]['beta_init'])
        self.critic_net_learnrate_init = float(config[self.env]['critic_net_learnrate_init'])
        self.critic_net_learnrate_decay = float(config[self.env]['critic_net_learnrate_decay'])
        self.critic_net_clipnorm = float(config[self.env]['critic_net_clipnorm'])
        self.actor_net_learnrate_init = float(config[self.env]['actor_net_learnrate_init'])
        self.actor_net_learnrate_decay = float(config[self.env]['actor_net_learnrate_decay'])
        self.actor_net_clipnorm = float(config[self.env]['actor_net_clipnorm'])
        self.train_hist_file = config[self.env]['train_hist_file']
        self.critic_model_file = config[self.env]['critic_model_file']
        self.actor_model_file = config[self.env]['actor_model_file']

        # parameters
        self.total_steps = 0
        self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
        self.actor_net_learnrate = self.actor_net_learnrate_init * (self.actor_net_learnrate_decay ** self.total_steps)

    def get_action(self, state, critic_model):
        q = critic_model.predict(np.array(state).reshape((1, self.greedysnake.SIZE ** 2)))
        sm = np.array(tf.nn.softmax(q)).reshape((4))
        rand = np.random.rand()
        action = None
        if 0 <= rand < sm[0]:
            action = Direction.UP
        elif sm[0] <= rand < sm[0] + sm[1]:
            action = Direction.DOWN
        elif sm[0] + sm[1] <= rand < sm[0] + sm[1] + sm[2]:
            action = Direction.LEFT
        elif sm[0] + sm[1] + sm[2] <= rand <= 1.0:
            action = Direction.RIGHT
        return action, q, sm

    def get_action_index(self, action):
        if action == Direction.UP:
            return 0
        elif action == Direction.DOWN:
            return 1
        elif action == Direction.LEFT:
            return 2
        elif action == Direction.RIGHT:
            return 3
        
    def get_nfq(self):

        # critic layers
        critic_model = keras.Sequential([
            keras.layers.Input(shape = (self.greedysnake.SIZE ** 2)), 
            keras.layers.Dense(50, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(50, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(4, activation = 'tanh', kernel_initializer='glorot_normal')
        ], name = 'critic')   

        # optimizer
        c_opt = keras.optimizers.SGD(
            lr = self.critic_net_learnrate, 
            clipnorm = self.critic_net_clipnorm
        )

        # critic model
        critic_model.compile(loss = keras.losses.MSE, optimizer = c_opt)

        return critic_model


    def get_state(self):
        display = ''
        frame = np.zeros(shape=(self.greedysnake.SIZE, self.greedysnake.SIZE), dtype=np.float32)
        # generate states for N(s, a)
        for i in range(self.greedysnake.SIZE ** 2):
            row = i // self.greedysnake.SIZE
            col = i % self.greedysnake.SIZE
            snake_index = self.greedysnake.is_snake(row, col)

            # snake
            if snake_index > -1:

                # snake head
                if snake_index == 0: 
                    frame[row, col] = 0.5
                    display += '@'

                # snake body
                else:
                    frame[row, col] = 0.3
                    display += 'O'

            # food
            elif (np.array([row, col]) == self.greedysnake.food).all():
                frame[row, col] = 1.0
                display += '#'
            
            # block
            else: 
                frame[row, col] = 0.1
                display += '-'

            # switch line
            if col == self.greedysnake.SIZE - 1:
                display += '\n'
        return frame, display
        
    def run(self):
        
        # define deep learning network
        critic_model = self.get_nfq()
        
        # statics
        scores = []
        hits = 0
        eats = 0

        for e in range(self.max_epochs):

            # execute steps for greedy snake
            s_arr = []
            s_a_t_add_1_arr = []
            r_arr = []
            t_arr = []
            q_arr = []

            # buffer
            s_t_temp = None
            a_t_temp = None
            
            # start steps
            for i in range(self.max_steps):

                # observe state and action at t = 0
                if i == 0:
                    s_t = self.get_state()[0].reshape((1, self.greedysnake.SIZE ** 2))
                    a_t = self.get_action(s_t, critic_model)[0]
                else: 
                    s_t = s_t_temp
                    a_t = a_t_temp
                s_arr.append(s_t)

                # take action via eps greedy, get reward
                signal = self.greedysnake.step(a_t)
                r = None

                # signal reward
                if signal == Signal.HIT:
                    r = -1
                    hits += 1
                elif signal == Signal.EAT:
                    r = 1
                    eats += 1
                elif signal == Signal.NORMAL:
                    r = 0.1

                r_arr.append(r)

                # observe state after action
                s_t = np.copy(s_t) #backup s_t
                display = self.get_state()[1]
                s_t_add_1 = self.get_state()[0].reshape((1, self.greedysnake.SIZE ** 2))
                s_t_temp = s_t_add_1
                s_a_t_add_1_arr.append(s_t_add_1)
                
                # choose action at t+1
                gares = self.get_action(s_t_add_1, critic_model)
                a_t_add_1 = gares[0]
                a_t_temp = a_t_add_1

                # get teacher for critic net (online learning)
                q_t = critic_model.predict(s_t)
                q_t_add_1_max = np.amax(np.array(critic_model.predict(s_t_add_1)))
                t = [0,0,0,0]
                for j in range(len(t)):
                    if j == self.get_action_index(a_t):
                        t[j] = r + self.gamma * q_t_add_1_max
                        if r == -1:
                            t[j] = r
                    else:
                        t[j] = np.array(q_t).reshape((4))[j]
                q_arr.append(q_t)
                t_arr.append(t)

                # accumulate index
                self.total_steps += 1

                # update learn rate and eps
                self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
                self.actor_net_learnrate = self.actor_net_learnrate_init * (self.actor_net_learnrate_decay ** self.total_steps)
                K.set_value(critic_model.optimizer.learning_rate, self.critic_net_learnrate)

                # display information
                a_print = str(a_t_add_1)
                r_print = str(float(r))
                t_print = str(np.array(t))
                predict_print = str(q_t)
                diff_print = str(abs(t - q_t))

                # calc stats
                if len(scores) < 1000:
                    scores.append(len(self.greedysnake.snake))
                else:
                    scores.pop(0)
                    scores.append(len(self.greedysnake.snake))
                avg = sum(scores) / len(scores)

                # print to debug
                print('Step = ' + str(i) + ' / Epoch = ' + str(e) + ' / Total Steps = ' + str(self.total_steps))
                print('action = ' + a_print + ' / reward = ' + r_print)
                print('teacher(Q) = ' + t_print + ' / predict(Q) = ' + predict_print +' / diff = ' + diff_print)
                print('Thousand steps average score = ' + str(avg))
                print('Hit rate = ' + str(hits / self.total_steps))
                print('Eat rate = ' + str(eats / self.total_steps))
                print(display)
                print(gares[1])
                
            # train steps
            s = np.array(s_arr, dtype=np.float32).reshape((len(s_arr), self.greedysnake.SIZE**2))
            t = np.array(t_arr, dtype=np.float32).reshape((len(t_arr), 4))
            critic_model.fit(s, t, epochs=self.critic_net_epochs, verbose=1, batch_size = self.batch_size)


            # record train history
            #f.write(str(critic_hist.history)+'\n')
            #f.write(str(actor_hist.history)+'\n')
            #f.close()

            # save model to file
            #critic_model.save(self.critic_model_file)
            #actor.save(self.actor_model_file) # BUG saving subclass model actor not succeed


if __name__ == "__main__":
    d = Driver()
    d.run()
        
