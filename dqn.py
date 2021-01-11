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


class Target(keras.Model):

    def __init__(self, critic):
        config = configparser.ConfigParser()
        config.read('dqn.ini')
        self.env = config['ENV']['env']
        self.gamma = float(config[self.env]['gamma'])
        super(Target, self).__init__()
        self.critic = critic
        self.target = keras.models.clone_model(critic)
        self.batch_size = int(config[self.env]['batch_size'])

    def compile(self, optimizer, loss):
        super(Target, self).compile()
        self.target_optimizer = optimizer
        self.loss = loss

    def train_step(self, data):

        state_t_add_1, q, reward = data[0]

        # train target
        with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
            tape.watch(self.target.trainable_weights)
            ts_ = self.target(state_t_add_1)
            y = reward + self.gamma * ts_ - q
            t = np.zeros((self.batch_size, 1))                                                        
            target_loss = self.loss(t, y)
        target_grads = tape.gradient(target_loss, self.target.trainable_weights)

        #print('============= test gradient ===================')
        #tf.print(tape.gradient(target_loss, y))

        self.target_optimizer.apply_gradients(
            zip(target_grads, self.target.trainable_weights)
        )
        return {"TargetLoss": target_loss}

    def call(self, state):
        return self.target(state)

    def predict_target(self, state):
        return self.target(state)


class Driver:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('dqn.ini')
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

    def get_action(self, state, critic_model):

        # get q values for all actions and compare the max q value
        # action 1 (UP)
        state = state.reshape((self.greedysnake.SIZE ** 2))
        action = np.zeros((4))

        action[0] = 1
        action[1] = 0
        action[2] = 0
        action[3] = 0
        state_action = np.array(tf.concat([state, action], axis = 0)).reshape(1, self.greedysnake.SIZE ** 2 + 4)
        qvalue_a1 = critic_model.predict(state_action)


        # action 2 (DOWN)
        action[0] = 0
        action[1] = 1
        action[2] = 0
        action[3] = 0
        state_action = np.array(tf.concat([state, action], axis = 0)).reshape(1, self.greedysnake.SIZE ** 2 + 4)
        qvalue_a2 = critic_model.predict(state_action)

        # action 3 (LEFT)
        action[0] = 0
        action[1] = 0
        action[2] = 1
        action[3] = 0
        state_action = np.array(tf.concat([state, action], axis = 0)).reshape(1, self.greedysnake.SIZE ** 2 + 4)
        qvalue_a3 = critic_model.predict(state_action)

        # action 4 (RIGHT)
        action[0] = 0
        action[1] = 0
        action[2] = 0
        action[3] = 1
        state_action = np.array(tf.concat([state, action], axis = 0)).reshape(1, self.greedysnake.SIZE ** 2 + 4)
        qvalue_a4 = critic_model.predict(state_action)

        # figure out the best legal action
        q_arr = []
        act_arr = []
        dict = {Direction.UP: float(qvalue_a1), Direction.DOWN: float(qvalue_a2), 
                Direction.LEFT: float(qvalue_a3), Direction.RIGHT: float(qvalue_a4)}
        sorted_dict = {k: v for k, v in sorted(dict.items(), reverse=True, key=lambda item: item[1])}
        for key in sorted_dict:
            q_arr.append(sorted_dict[key])
            act_arr.append(key)
        return act_arr[0], q_arr[0], action
        
    def get_dqn(self):

        # critic layers
        critic_model = keras.Sequential([
            keras.layers.Input(shape = (self.greedysnake.SIZE ** 2 + 4)), 
            #keras.layers.Dense(1024, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(1024, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(512, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            #keras.layers.Dense(512, activation = 'relu', kernel_initializer='glorot_normal'),
            #keras.layers.BatchNormalization(),
            keras.layers.Dense(100, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(100, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(50, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(50, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation = 'tanh', kernel_initializer='glorot_normal')
        ], name = 'critic')   

        # optimizer
        c_opt = keras.optimizers.SGD(
            lr = self.critic_net_learnrate, 
            clipnorm = self.critic_net_clipnorm
        )
        t_opt = keras.optimizers.SGD(
            lr = self.critic_net_learnrate, 
            clipnorm = self.critic_net_clipnorm
        )
        
        # critic model
        critic_model.compile(loss = keras.losses.MSE, optimizer = c_opt)

        # target model
        target = Target(critic=critic_model)
        target.target.set_weights(critic_model.get_weights())
        target.compile(loss = keras.losses.MSE, optimizer = t_opt)

        return critic_model, target


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
        critic_model, target = self.get_dqn()
        
        # statics
        scores = []
        hits = 0
        eats = 0

        for e in range(self.max_epochs):

            # execute steps for greedy snake
            s_arr = []
            s_a_t_add_1_arr = []
            s_a_arr = []
            r_arr = []
            t_arr = []
            q_arr = []

            # buffer
            s_t_temp = None
            a_t_temp = None
            a_t_map_temp = None
            
            # start steps
            for i in range(self.max_steps):

                # observe state and action at t = 0
                if i == 0:
                    s_t = self.get_state()[0].reshape((1, self.greedysnake.SIZE ** 2))
                    gares = self.get_action(s_t, critic_model)
                    a_t = gares[0]
                    a_t_map = gares[2].reshape((1, 4))
                else: 
                    s_t = s_t_temp
                    a_t = a_t_temp
                    a_t_map = a_t_map_temp
                s_a_t = tf.concat([s_t, a_t_map], axis=1)
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

                r_arr.append(r)

                # observe state after action
                s_t = np.copy(s_t) #backup s_t
                display = self.get_state()[1]
                s_t_add_1 = self.get_state()[0].reshape((1, self.greedysnake.SIZE ** 2))
                s_t_temp = s_t_add_1
                
                # choose action at t+1
                gares = self.get_action(s_t_add_1, critic_model)
                a_t_add_1 = gares[0]
                a_t_add_1_map = gares[2].reshape((1, 4))
                a_t_temp = a_t_add_1
                a_t_map_temp = a_t_add_1_map

                # get teacher for critic net (online learning)
                s_a_t_add_1 = np.array(tf.concat([s_t_add_1, a_t_add_1_map], axis=1))
                s_a_t_add_1_arr.append(s_a_t_add_1)
                q_t = critic_model.predict(s_a_t)
                target_sa = target.predict(s_a_t_add_1)
                t = r + self.gamma * target_sa
                if r == -1:
                    t = r
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
            s_a = np.array(s_a_arr, dtype=np.float32).reshape((len(s_a_arr), self.greedysnake.SIZE**2 + 4))
            s_a_ = np.array(s_a_t_add_1_arr, dtype=np.float32).reshape((len(s_a_t_add_1_arr), self.greedysnake.SIZE**2 + 4))
            t = np.array(t_arr, dtype=np.float32).reshape((len(t_arr), 1))
            q = np.array(q_arr, dtype=np.float32).reshape((len(q_arr), 1))
            r = np.array(r_arr, dtype=np.float32).reshape((len(r_arr), 1))
            critic_model.fit(s_a, t, epochs=self.critic_net_epochs, verbose=1, batch_size = self.batch_size)
            target.fit([s_a_, q, r], epochs=self.actor_net_epochs, verbose=1, batch_size = self.batch_size)

            if e % 20 == 0:
                target.target.set_weights(critic_model.get_weights())

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
        
