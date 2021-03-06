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
from collections import deque
import copy
import sys
import warnings
import math
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)

class ADHDP(keras.Model):

    def __init__(self, critic, actor):
        config = configparser.ConfigParser()
        config.read('adhdp_cnn.ini')
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
            action_map = self.actor(state)
            state_action = tf.concat([state, action_map], 3)
            q = self.critic(state_action)
            t = np.ones((self.batch_size, 1))              
            t.fill(30.333333)                                             
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

    def save_models(self):
        self.critic.save('adhdp_cnn_critic')
        self.actor.save('adhdp_cnn_actor')



class Driver:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('adhdp_cnn.ini')
        self.env = config['ENV']['env']
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.max_epochs = int(config[self.env]['max_epochs'])
        self.max_steps = int(config[self.env]['max_steps'])
        self.batch_size = int(config[self.env]['batch_size'])
        self.memory_size = int(config[self.env]['memory_size'])
        self.mini_batch_size = int(config[self.env]['mini_batch_size'])
        self.critic_net_epochs = int(config[self.env]['critic_net_epochs'])
        self.actor_net_epochs = int(config[self.env]['actor_net_epochs'])
        self.gamma = float(config[self.env]['gamma'])
        self.beta_init = float(config[self.env]['beta_init'])
        self.beta_decay = float(config[self.env]['beta_decay'])
        self.critic_net_learnrate_init = float(config[self.env]['critic_net_learnrate_init'])
        self.critic_net_learnrate_decay = float(config[self.env]['critic_net_learnrate_decay'])
        self.critic_net_clipnorm = float(config[self.env]['critic_net_clipnorm'])
        self.actor_net_learnrate_init = float(config[self.env]['actor_net_learnrate_init'])
        self.actor_net_learnrate_decay = float(config[self.env]['actor_net_learnrate_decay'])
        self.actor_net_clipnorm = float(config[self.env]['actor_net_clipnorm'])
        self.train_hist_file = config[self.env]['train_hist_file']
        self.critic_model_file = config[self.env]['critic_model_file']
        self.actor_model_file = config[self.env]['actor_model_file']
        self.timeslip_size = int(config[self.env]['timeslip_size'])
        self.timeslip = np.zeros(shape=(self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size))

        # parameters
        self.total_steps = 0
        self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
        self.actor_net_learnrate = self.actor_net_learnrate_init * (self.actor_net_learnrate_decay ** self.total_steps)
        self.beta = self.beta_init * (self.beta_decay ** self.total_steps)


    def random_action_map(self):
        rand = random.randint(0, 3)
        action_map = np.zeros((self.greedysnake.SIZE, self.greedysnake.SIZE))
        central = self.greedysnake.SIZE // 2
        if rand == 0.0:
            action_map[0, central] = 1.0
        elif rand == 1.0:
            action_map[self.greedysnake.SIZE-1, central] = 1.0
        elif rand == 2.0:
            action_map[central, 0] = 1.0
        elif rand == 3.0:
            action_map[central, self.greedysnake.SIZE - 1] = 1.0
        return action_map

    def get_action(self, action_map):
        map = np.array(action_map).reshape((self.greedysnake.SIZE, self.greedysnake.SIZE))
        action = None
        head_row = self.greedysnake.snake[0][0]
        head_col = self.greedysnake.snake[0][1]
        map_up = None
        map_down = None
        map_left = None
        map_right = None

        # impact prevention
        if head_row == 0:
            map_up = -math.inf
        else:
            map_up = map[(head_row - 1), head_col]
        if head_row == self.greedysnake.SIZE - 1:
            map_down = -math.inf
        else:
            map_down = map[(head_row + 1), head_col]
        if head_col == 0:
            map_left = -math.inf
        else:
            map_left = map[(head_row), head_col - 1]
        if head_col == self.greedysnake.SIZE - 1:
            map_right = -math.inf
        else:
            map_right = map[(head_row), head_col + 1]        
        map_values = np.array([map_up, map_down, map_left, map_right])
        argmax = np.argmax(map_values)
        if argmax == 0:
            action = Direction.UP
        elif argmax == 1:
            action = Direction.DOWN
        elif argmax == 2:
            action = Direction.LEFT
        elif argmax == 3:
            action = Direction.RIGHT
        return action, map
        

    def get_adhdp(self):

        initializer = keras.initializers.HeNormal()

        # critic layers
        critic_model = keras.Sequential([
            keras.layers.Input(shape = (self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size + 1)), 
            keras.layers.Conv2D(
                16, (2, 2), 
                padding='same', 
                activation='relu', 
                kernel_initializer=initializer, 
            ),
            keras.layers.Conv2D(
                32, (1, 1), 
                padding='same', 
                activation='relu', 
                kernel_initializer=initializer, 
            ),
            keras.layers.Conv2D(
                64, (3, 3), 
                padding='same', 
                activation='relu', 
                kernel_initializer=initializer, 
            ),
            keras.layers.Conv2D(
                64, (3, 3), 
                padding='same', 
                activation='relu', 
                kernel_initializer=initializer, 
            ),
            keras.layers.Conv2D(
                64, (3, 3), 
                padding='same', 
                activation='relu', 
                kernel_initializer=initializer, 
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(500, activation = 'relu', kernel_initializer=initializer),
            keras.layers.Dense(200, activation = 'relu', kernel_initializer=initializer),
            keras.layers.Dense(100, activation = 'relu', kernel_initializer=initializer),
            keras.layers.Dense(1, kernel_initializer=initializer)
        ], name = 'critic')

        # actor layers
        actor_model = keras.Sequential([
            keras.layers.Input(shape = (self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size)), 
            keras.layers.Conv2D(
                16, (2, 2), 
                padding='same', 
                activation='relu', 
                kernel_initializer=initializer, 
            ),
            keras.layers.Conv2D(
                32, (1, 1), 
                padding='same', 
                activation='relu', 
                kernel_initializer=initializer, 
            ),
            keras.layers.Conv2D(
                64, (3, 3), 
                padding='same', 
                activation='relu', 
                kernel_initializer=initializer, 
            ),
            keras.layers.Conv2D(
                64, (3, 3), 
                padding='same', 
                activation='relu', 
                kernel_initializer=initializer, 
            ),
            keras.layers.Conv2D(
                1, (1, 1),
                padding='same',
                kernel_initializer=initializer, 
            ), 
        ], name = 'actor')        

        # optimizer
        c_opt = keras.optimizers.Adam(
            lr = self.critic_net_learnrate, 
            clipnorm = self.critic_net_clipnorm
        )
        a_opt = keras.optimizers.Adam(
            lr = self.actor_net_learnrate, 
            clipnorm = self.actor_net_clipnorm
        )

        # models
        critic_model.compile(loss = keras.losses.MSE, optimizer = c_opt)

        # actor model
        adhdp = ADHDP(critic=critic_model, actor=actor_model)
        adhdp.compile(loss = keras.losses.MSE, optimizer = a_opt) # loss is MSE to compare the Q values
        return critic_model, adhdp

    def write_to_timeslip(self):
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
                frame[row, col] = 0.
                display += '-'

            # switch line
            if col == self.greedysnake.SIZE - 1:
                display += '\n'
            # store frame to timeslip

        self.timeslip = np.insert(self.timeslip, 0, frame, axis=2)
        self.timeslip = np.delete(self.timeslip, self.timeslip_size, axis=2)
        return display
        
    def run(self):

        # record random initial steps
        for i in range(self.timeslip_size + 1):
            ram = self.random_action_map()
            a = self.get_action(ram)[0]
            self.greedysnake.step(a)
            display = self.write_to_timeslip()
            print('=========Initial Steps===========')
            print(display)
        
        # define deep learning network
        critic_model, adhdp = self.get_adhdp()
        
        # statics
        scores = deque(maxlen=1000)
        max_score = 0
        hits = 0
        eats = 0


        for e in range(self.max_epochs):

            # execute steps for greedy snake
            s_memory = deque()
            s_a_memory = deque()
            r_memory = deque()
            t_memory = deque()

            # buffer
            s_current_temp = None
            a_current_temp = None
            actmap_current_temp = None
            
            # start steps
            i = 0
            stamina = 0
            stamina_max = self.greedysnake.SIZE
            for i in range(self.max_steps):

                # observe state and action at t = 0
                if i == 0:
                    s_current = self.timeslip
                    actmap_current = adhdp.predict_actor(s_current.reshape(1, self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size))
                    a_current = self.get_action(np.array(actmap_current).reshape(self.greedysnake.SIZE, self.greedysnake.SIZE))[0]
                else: 
                    s_current = s_current_temp
                    a_current = a_current_temp
                    actmap_current = actmap_current_temp
                s_a_current = tf.concat([s_current, np.array(actmap_current).reshape((self.greedysnake.SIZE, self.greedysnake.SIZE, 1))], axis=2)

                s_memory.append(s_current)
                s_a_memory.append(s_a_current)

                # take action and fetch reward
                signal = self.greedysnake.step(a_current)
                r = None
                if signal == Signal.HIT:
                    r = -1
                    stamina = 0
                    hits += 1
                elif signal == Signal.EAT:
                    r = 1
                    stamina = stamina_max
                    eats += 1
                elif signal == Signal.NORMAL:
                    stamina -= 1
                    if stamina < 0:
                        stamina = 0
                    r = stamina / stamina_max
                r_memory.append(r)

                # observe state after action
                display = self.write_to_timeslip()
                s_future = self.timeslip
                s_current_temp = s_future
                
                # choose action at t+1
                actmap_future = adhdp.predict_actor(np.array(s_future).reshape(1, self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size))
                get_action_result = self.get_action(np.array(actmap_future).reshape(self.greedysnake.SIZE, self.greedysnake.SIZE))
                a_future = get_action_result[0]
                actmap_current_temp = actmap_future
                a_current_temp = a_future

                # get teacher for critic net
                s_a_future = tf.concat([s_future, actmap_future[0,:,:,:]], axis=2)
                q_current = critic_model.predict(np.array(s_a_current).reshape(1, self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size + 1))
                q_future = critic_model.predict(np.array(s_a_future).reshape(1, self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size + 1))
                t = r + self.gamma * q_future
                if signal == Signal.HIT:
                    t = r
                t_memory.append(t)

                # accumulate index
                self.total_steps += 1

                # update learn rate and eps
                self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
                self.actor_net_learnrate = self.actor_net_learnrate_init * (self.actor_net_learnrate_decay ** self.total_steps)
                self.beta = self.beta_init * (self.beta_decay ** self.total_steps)
                K.set_value(critic_model.optimizer.learning_rate, self.critic_net_learnrate)
                K.set_value(adhdp.optimizer.learning_rate, self.actor_net_learnrate)

                # display information
                a_print = str(a_future)
                r_print = str(float(r))
                t_print = str(np.array(t))
                predict_print = str(q_current)
                diff_print = str(abs(t - q_current))

                # calc stats
                scores.append(len(self.greedysnake.snake))
                avg = sum(scores) / len(scores)
                if avg > max_score:
                    max_score = avg

                # print to debug
                print('Step = ' + str(i) + ' / Epoch = ' + str(e) + ' / Total Steps = ' + str(self.total_steps))
                print('action = ' + a_print + ' / reward = ' + r_print)
                print('teacher(Q) = ' + t_print + ' / predict(Q) = ' + predict_print +' / diff = ' + diff_print)
                print('Thousand steps average score = ' + str(avg))
                print('Max average score = ' + str(max_score))
                print('Hit rate = ' + str(hits / self.total_steps))
                print('Eat rate = ' + str(eats / self.total_steps))
                print(display)
                #print(get_action_result[1])

            # train steps
            s = np.array(list(s_memory), dtype=np.float32).reshape((len(list(s_memory)), self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size))
            s_a = np.array(list(s_a_memory), dtype=np.float32).reshape((len(list(s_a_memory)), self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size + 1))
            t = np.array(list(t_memory), dtype=np.float32).reshape((len(list(t_memory)), 1))
            critic_model.fit(s_a, t, epochs=self.critic_net_epochs, verbose=0, batch_size = self.batch_size)
            adhdp.fit(s, t, epochs=self.actor_net_epochs, verbose=0, batch_size = self.batch_size)

            # save model to file
            adhdp.save_models()



if __name__ == "__main__":
    d = Driver()
    d.run()
