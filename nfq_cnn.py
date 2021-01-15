from greedysnake import GreedySnake, Direction, Signal
import time
import numpy as np
from threading import Thread
import subprocess
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from collections import OrderedDict
from collections import deque
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
        config.read('nfq_cnn.ini')
        self.env = config['ENV']['env']
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.max_epochs = int(config[self.env]['max_epochs'])
        self.max_steps = int(config[self.env]['max_steps'])
        self.epsilon_init = float(config[self.env]['epsilon_init'])
        self.epsilon_decay = float(config[self.env]['epsilon_decay'])
        self.batch_size = int(config[self.env]['batch_size'])
        self.memory_size = int(config[self.env]['memory_size'])
        self.mini_batch_size = int(config[self.env]['mini_batch_size'])
        self.critic_net_epochs = int(config[self.env]['critic_net_epochs'])
        self.gamma = float(config[self.env]['gamma'])
        self.beta_init = float(config[self.env]['beta_init'])
        self.beta_decay = float(config[self.env]['beta_decay'])
        self.critic_net_learnrate_init = float(config[self.env]['critic_net_learnrate_init'])
        self.critic_net_learnrate_decay = float(config[self.env]['critic_net_learnrate_decay'])
        self.critic_net_clipnorm = float(config[self.env]['critic_net_clipnorm'])
        self.train_hist_file = config[self.env]['train_hist_file']
        self.critic_model_file = config[self.env]['critic_model_file']

        # parameters
        self.total_steps = 0
        self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
        self.beta = self.beta_init * (self.beta_decay ** self.total_steps)
        self.epsilon = self.epsilon_init * (self.epsilon_decay ** self.total_steps)


    '''
    def get_action(self, state, critic_model, epsilon):

        rand_strategy = np.random.rand()

        # random action
        if 0 <= rand_strategy <= epsilon:
            q = critic_model.predict(np.array(state).reshape((1, self.greedysnake.SIZE, self.greedysnake.SIZE, 3)))
            sm = np.array(tf.nn.softmax(q)).reshape((4))
            rand = np.random.randint(0, 4)
            action = None
            if rand == 0:
                action = Direction.UP
            elif rand == 1:
                action = Direction.DOWN
            elif rand == 2:
                action = Direction.LEFT
            elif rand == 3:
                action = Direction.RIGHT
            return action, q, sm

        # greedy
        else:
            q = critic_model.predict(np.array(state).reshape((1, self.greedysnake.SIZE, self.greedysnake.SIZE, 3)))
            sm = np.array(tf.nn.softmax(q)).reshape((4))
            q_np = np.array(q).reshape((4))
            argmax = np.argmax(q_np)
            action = None
            if argmax == 0:
                action = Direction.UP
            elif argmax == 1:
                action = Direction.DOWN
            elif argmax == 2:
                action = Direction.LEFT
            elif argmax == 3:
                action = Direction.RIGHT
            return action, q, sm
    '''

    def get_action(self, state, critic_model, epsilon):

        rand_strategy = np.random.rand()

        # greedy algorithmus
        if 0 <= rand_strategy <= epsilon:
            q = critic_model.predict(np.array(state).reshape((1, self.greedysnake.SIZE, self.greedysnake.SIZE, 3)))
            sm = np.array(tf.nn.softmax(q)).reshape((4))
            action = None
            food_smell_map = np.array(state)[:,:,:,2].reshape((self.greedysnake.SIZE, self.greedysnake.SIZE))
            smells = [0.,0.,0.,0.]
            for i in range(self.greedysnake.SIZE ** 2):
                row = i // self.greedysnake.SIZE
                col = i % self.greedysnake.SIZE
                snake_index = self.greedysnake.is_snake(row, col)
                # snake head
                if snake_index == 0:
                    try:
                        smells[0] = food_smell_map[row-1, col]
                        smells[1] = food_smell_map[row+1, col]
                        smells[2] = food_smell_map[row, col-1]
                        smells[3] = food_smell_map[row-1, col+1]
                    except IndexError:
                        pass
            argmax = np.argmax(np.array(smells))
            if argmax == 0:
                if self.greedysnake.head_direction != Direction.DOWN:
                    action = Direction.UP
                else:
                    action = Direction.RIGHT
            elif argmax == 1:
                if self.greedysnake.head_direction != Direction.UP:
                    action = Direction.DOWN
                else:
                    action = Direction.LEFT
            elif argmax == 2:
                if self.greedysnake.head_direction != Direction.RIGHT:
                    action = Direction.LEFT
                else:
                    action = Direction.UP
            elif argmax == 3:
                if self.greedysnake.head_direction != Direction.LEFT:
                    action = Direction.RIGHT
                else:
                    action = Direction.DOWN
            return action, q, sm

        # greedy
        else:
            q = critic_model.predict(np.array(state).reshape((1, self.greedysnake.SIZE, self.greedysnake.SIZE, 3)))
            sm = np.array(tf.nn.softmax(q)).reshape((4))
            q_np = np.array(q).reshape((4))
            argmax = np.argmax(q_np)
            action = None
            if argmax == 0:
                action = Direction.UP
            elif argmax == 1:
                action = Direction.DOWN
            elif argmax == 2:
                action = Direction.LEFT
            elif argmax == 3:
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
            keras.layers.Input(shape = (self.greedysnake.SIZE, self.greedysnake.SIZE, 3)), 
            keras.layers.Conv2D(
                9, (5, 5), 
                padding='same', 
                activation='relu', 
                kernel_initializer='random_normal', 
            ),
            keras.layers.Conv2D(
                9, (3, 3), 
                padding='same', 
                activation='relu', 
                kernel_initializer='random_normal', 
            ),
            keras.layers.Conv2D(
                9, (3, 3), 
                padding='same', 
                activation='relu', 
                kernel_initializer='random_normal', 
            ),
            keras.layers.Conv2D(
                9, (3, 3), 
                padding='same', 
                activation='relu', 
                kernel_initializer='random_normal', 
            ),
            keras.layers.Conv2D(
                9, (3, 3), 
                padding='same', 
                activation='relu', 
                kernel_initializer='random_normal', 
            ),
            keras.layers.Conv2D(
                9, (1, 1), 
                padding='same', 
                activation='relu', 
                kernel_initializer='random_normal', 
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(1600, activation = 'relu', kernel_initializer='random_normal'),
            keras.layers.Dense(800, activation = 'relu', kernel_initializer='random_normal'),
            keras.layers.Dense(400, activation = 'relu', kernel_initializer='random_normal'),
            keras.layers.Dense(100, activation = 'relu', kernel_initializer='random_normal'),
            keras.layers.Dense(4, kernel_initializer='random_normal')
        ], name = 'critic')

        # optimizer
        c_opt = keras.optimizers.Adam(
            lr = self.critic_net_learnrate, 
            clipnorm = self.critic_net_clipnorm
        )

        # critic model
        critic_model.compile(loss = keras.losses.MSE, optimizer = c_opt)

        return critic_model


    def get_state(self):
        display = ''
        frame_head = np.zeros(shape=(self.greedysnake.SIZE, self.greedysnake.SIZE, 1), dtype=np.float32)
        frame_body = np.zeros(shape=(self.greedysnake.SIZE, self.greedysnake.SIZE, 1), dtype=np.float32)
        frame_food = np.zeros(shape=(self.greedysnake.SIZE, self.greedysnake.SIZE, 1), dtype=np.float32)
        # generate states for N(s, a)
        for i in range(self.greedysnake.SIZE ** 2):
            row = i // self.greedysnake.SIZE
            col = i % self.greedysnake.SIZE
            snake_index = self.greedysnake.is_snake(row, col)

            # snake
            if snake_index > -1:

                # snake head
                if snake_index == 0: 
                    frame_head[row, col] = 1.
                    display += '@'

                # snake body
                else:
                    frame_body[row, col] = 1.
                    display += 'O'

            # food
            elif (np.array([row, col]) == self.greedysnake.food).all():
                frame_food[row, col] = 1.
                display += '#'
            
            # block
            else: 
                display += '-'

            # switch line
            if col == self.greedysnake.SIZE - 1:
                display += '\n'

        # concat frames
        frame = np.concatenate((frame_head, frame_body, frame_food), axis=2)
            
        return frame, display
        
    def run(self):
        
        # define deep learning network
        critic_model = self.get_nfq()
        
        # statics
        scores = []
        hits = 0
        eats = 0

        # database
        s_memory = deque(maxlen=self.memory_size)
        s_a_future_memory = deque(maxlen=self.memory_size)
        r_memory = deque(maxlen=self.memory_size)
        t_memory = deque(maxlen=self.memory_size)
        q_memory = deque(maxlen=self.memory_size)

        for e in range(self.max_epochs):

            # buffer
            s_current_temp = None
            a_current_temp = None
            
            # start steps
            i = 0
            while i < self.max_steps:

                s_current = None
                a_current = None

                # observe state and action at t = 0
                if i == 0:
                    s_current = self.get_state()[0].reshape((1, self.greedysnake.SIZE, self.greedysnake.SIZE, 3))
                    a_current = self.get_action(s_current, critic_model, self.epsilon)[0]
                else: 
                    s_current = s_current_temp
                    a_current = a_current_temp
                s_memory.append(s_current)
                display = self.get_state()[1]

                # take action via eps greedy, get reward
                signal = self.greedysnake.step(a_current)
                r = None

                # signal reward
                if signal == Signal.HIT:
                    r = -1.
                    hits += 1
                    # i = self.max_steps - 1    #  learn on hit
                elif signal == Signal.EAT:
                    r = 1.
                    eats += 1
                elif signal == Signal.NORMAL:
                    r = 0.
                r_memory.append(r)

                # observe state after action
                s_future = self.get_state()[0].reshape((1, self.greedysnake.SIZE, self.greedysnake.SIZE, 3))
                s_current_temp = s_future
                s_a_future_memory.append(s_future)
                
                # choose action at t+1
                get_action_result = self.get_action(s_future, critic_model, self.epsilon)
                a_future = get_action_result[0]
                a_current_temp = a_future

                # get teacher for critic net (online learning)
                q_current = critic_model.predict(s_current)
                q_future_max = np.amax(np.array(critic_model.predict(s_future)))
                t = [0,0,0,0]
                for j in range(len(t)):
                    if j == self.get_action_index(a_current):
                        q_temp = np.array(q_current).reshape((4))[j]
                        t[j] = q_temp + self.beta_init * (r + self.gamma * q_future_max - q_temp)
                        if signal == Signal.HIT:
                            t[j] = q_temp + self.beta_init * (r + self.gamma * 0 - q_temp)
                    else:
                        t[j] = np.array(q_current).reshape((4))[j]
                q_memory.append(q_current)
                t_memory.append(t)

                # accumulate index
                self.total_steps += 1

                # update learn rate and eps
                self.epsilon = self.epsilon_init * (self.epsilon_decay ** self.total_steps)
                self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
                self.beta = self.beta_init * (self.beta_decay ** self.total_steps)
                K.set_value(critic_model.optimizer.learning_rate, self.critic_net_learnrate)

                # display information
                a_print = str(a_current)
                r_print = str(float(r))
                t_print = str(np.array(t))
                predict_print = str(q_current)
                diff_print = str(abs(t - q_current))

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
                print('teacher(Q) = ' + t_print + ' / predict(Q) = ' + predict_print +' / diff = ' + diff_print + ' / epsilon = ' + str(self.epsilon))
                print('Thousand steps average score = ' + str(avg))
                print('Hit rate = ' + str(hits / self.total_steps))
                print('Eat rate = ' + str(eats / self.total_steps))
                print(display)
                print(tf.nn.softmax(get_action_result[1]))

                # inc step counter
                i += 1
                
            # train steps
            mini_batch_size = self.mini_batch_size
            len_memory = len(list(s_memory))
            if len_memory < mini_batch_size:
                mini_batch_size = len_memory
            s_minibatch = random.sample(s_memory, mini_batch_size)
            t_minibatch = random.sample(t_memory, mini_batch_size)
            s = np.array(list(s_minibatch), dtype=np.float32).reshape((len(list(s_minibatch)), self.greedysnake.SIZE, self.greedysnake.SIZE, 3))
            t = np.array(list(t_minibatch), dtype=np.float32).reshape((len(t_minibatch), 4))
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
        
