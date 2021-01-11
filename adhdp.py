from greedysnake import GreedySnake, Direction, Signal
import time
import numpy as np
#import curses
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

class ADHDP(keras.Model):

    def __init__(self, critic, actor):
        config = configparser.ConfigParser()
        config.read('adhdp.ini')
        self.env = config['ENV']['env']
        super(ADHDP, self).__init__()
        self.critic = critic
        self.actor = actor
        self.batch_size = int(config[self.env]['batch_size'])

    def compile(self, optimizer, loss):
        super(ADHDP, self).compile()
        self.actor_optimizer = optimizer
        self.loss = loss

    def train_step(self, data):

        state, teacher_critic = data

        # train actor
        with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
            tape.watch(self.actor.trainable_weights)
            action = self.actor(state)
            state_action = tf.concat([state, action], 1)
            q = self.critic(state_action)
            t = np.ones((self.batch_size, 1))              
            t.fill(1.0)                                             
            actor_loss = self.loss(t, q)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)

        #print('============= test gradient ===================')
        #tf.print(tape.gradient(actor_loss, action_map))

        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_weights)
        )
        return {"ActorLoss": actor_loss}

    def call(self, state):
        return self.actor(state)

    def predict_actor(self, state):
        return self.actor(state)



class Driver:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('adhdp.ini')
        self.env = config['ENV']['env']
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.max_epochs = int(config[self.env]['max_epochs'])
        self.max_steps = int(config[self.env]['max_steps'])
        self.batch_size = int(config[self.env]['batch_size'])
        self.critic_net_epochs = int(config[self.env]['critic_net_epochs'])
        self.actor_net_epochs = int(config[self.env]['actor_net_epochs'])
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

    def print_action_softmax(self, action_map):
        sum_rows_arr = np.sum(action_map, axis=1)
        rand_row = np.random.rand()
        rows_prob = tf.nn.softmax(sum_rows_arr)
        sum_row = 0.
        row = None
        for i in range(np.array(rows_prob).shape[0]):
            if sum_row <= rand_row <= sum_row + rows_prob[i]:
                row = i
                break
            else:
                sum_row += rows_prob[i]
        column = action_map[row, :]
        col_prob = tf.nn.softmax(column)
        print(rows_prob)
        print(col_prob)

    def get_action(self, action_map):
        map = np.array(action_map).reshape((4))
        rand = np.random.rand()
        action = None
        if 0 <= rand < map[0]:
            action = Direction.UP
        elif map[0] <= rand < map[0] + map[1]:
            action = Direction.DOWN
        elif map[0] + map[1] <= rand < map[0] + map[1] + map[2]:
            action = Direction.LEFT
        elif map[0] + map[1] + map[2] <= rand <= 1.0:
            action = Direction.RIGHT
        return action, map
        
    def get_adhdp(self):

        # critic layers
        critic_model = keras.Sequential([
            keras.layers.Input(shape = (self.greedysnake.SIZE ** 2 + 4)), 
            keras.layers.Dense(self.greedysnake.SIZE ** 2 * 2, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2 * 2 , activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2 * 2 , activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2 * 2 , activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            keras.layers.Dense(self.greedysnake.SIZE ** 2 // 2, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation = 'tanh', kernel_initializer='glorot_normal')
        ], name = 'critic')

        # actor layers
        actor_model = keras.Sequential([
            keras.layers.Input(shape = (self.greedysnake.SIZE ** 2)), 
            keras.layers.Dense(self.greedysnake.SIZE ** 2 * 2, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2 * 2 , activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2 * 2 , activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2 * 2 , activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            keras.layers.Dense(self.greedysnake.SIZE ** 2 // 2, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(4, activation = 'softmax', kernel_initializer='glorot_normal'),
        ], name = 'actor')        

        # optimizer
        c_opt = keras.optimizers.SGD(
            lr = self.critic_net_learnrate, 
            clipnorm = self.critic_net_clipnorm
        )
        a_opt = keras.optimizers.SGD(
            lr = self.actor_net_learnrate, 
            clipnorm = self.actor_net_clipnorm
        )

        # models
        critic_model.compile(loss = keras.losses.MSE, optimizer = c_opt)

        # actor model
        adhdp = ADHDP(critic=critic_model, actor=actor_model)
        adhdp.compile(loss = keras.losses.MSE, optimizer = a_opt) # loss is MSE to compare the Q values
        return critic_model, adhdp


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
            # store frame to timeslip

        return frame, display
        
    def run(self):
        
        # define deep learning network
        critic_model, adhdp = self.get_adhdp()
        
        # statics
        scores = []
        hits = 0
        eats = 0


        for e in range(self.max_epochs):

            # execute steps for greedy snake
            s_arr = []
            s_a_arr = []
            r_arr = []
            t_arr = []

            # buffer
            s_t_temp = None
            a_t_temp = None
            actmap_t_temp = None
            
            # start steps
            for i in range(self.max_steps):

                # observe state and action at t = 0
                if i == 0:
                    s_t = self.get_state()[0].reshape((1, self.greedysnake.SIZE ** 2))
                    actmap_t = adhdp.predict_actor(s_t)
                    a_t = self.get_action(np.array(actmap_t).reshape(4))[0]
                else: 
                    s_t = s_t_temp
                    a_t = a_t_temp
                    actmap_t = actmap_t_temp
                s_a_t = tf.concat([s_t, actmap_t], axis=1)
                print('#############################################')

                s_arr.append(s_t)
                s_a_arr.append(s_a_t)

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

                # unvalid control reward (covers signal reward)
                cond0 = self.greedysnake.head_direction == Direction.LEFT and a_t == Direction.RIGHT
                cond1 = self.greedysnake.head_direction == Direction.RIGHT and a_t == Direction.LEFT
                cond2 = self.greedysnake.head_direction == Direction.UP and a_t == Direction.DOWN
                cond3 = self.greedysnake.head_direction == Direction.DOWN and a_t == Direction.UP
                if cond0 or cond1 or cond2 or cond3:
                    r = -1

                r_arr.append(r)

                # observe state after action
                s_t = np.copy(s_t) #backup s_t
                display = self.get_state()[1]
                s_t_add_1 = self.get_state()[0].reshape((1, self.greedysnake.SIZE ** 2))
                s_t_temp = s_t_add_1
                
                # choose action at t+1
                actmap_t_add_1 = adhdp.predict_actor(np.array(s_t_add_1).reshape(1, self.greedysnake.SIZE ** 2))
                gares = self.get_action(np.array(actmap_t_add_1).reshape(4))
                a_t_add_1 = gares[0]
                actmap_t_temp = actmap_t_add_1
                a_t_temp = a_t_add_1

                # get teacher for critic net (online learning)
                s_a_t_add_1 = np.array(tf.concat([s_t_add_1, actmap_t_add_1], axis=1))
                q_t = critic_model.predict(s_a_t)
                q_t_add_1 = critic_model.predict(s_a_t_add_1)
                t = r + self.gamma * q_t_add_1
                if r == -1:
                    t = r
                t_arr.append(t)

                # accumulate index
                self.total_steps += 1

                # update learn rate and eps
                self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
                self.actor_net_learnrate = self.actor_net_learnrate_init * (self.actor_net_learnrate_decay ** self.total_steps)
                K.set_value(critic_model.optimizer.learning_rate, self.critic_net_learnrate)
                K.set_value(adhdp.optimizer.learning_rate, self.actor_net_learnrate)

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
            s_a = np.array(s_a_arr, dtype=np.float32).reshape((len(s_a_arr), self.greedysnake.SIZE**2 + 4))
            t = np.array(t_arr, dtype=np.float32).reshape((len(t_arr), 1))
            critic_model.fit(s_a, t, epochs=self.critic_net_epochs, verbose=1, batch_size = self.batch_size)
            adhdp.fit(s, t, epochs=self.actor_net_epochs, verbose=1, batch_size = self.batch_size)

            # record train history
            #f.write(str(critic_hist.history)+'\n')
            #f.write(str(actor_hist.history)+'\n')
            #f.close()

            # save model to file
            #critic_model.save(self.critic_model_file)
            #adhdp.save(self.actor_model_file) # BUG saving subclass model adhdp not succeed


if __name__ == "__main__":
    d = Driver()
    d.run()
        
