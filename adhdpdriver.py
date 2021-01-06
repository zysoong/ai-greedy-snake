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
        config.read('adhdpdriver.ini')
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
            t.fill(1.0)                                             
            actor_loss = self.loss(t, q)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
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
        config.read('adhdpdriver.ini')
        self.env = config['ENV']['env']
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.max_epochs = int(config[self.env]['max_epochs'])
        self.max_steps = int(config[self.env]['max_steps'])
        self.batch_size = int(config[self.env]['batch_size'])
        self.critic_net_epochs = int(config[self.env]['critic_net_epochs'])
        self.actor_net_epochs = int(config[self.env]['actor_net_epochs'])
        self.gamma = float(config[self.env]['gamma'])
        self.epsilon_init = float(config[self.env]['epsilon_init'])
        self.epsilon_decay = float(config[self.env]['epsilon_decay'])
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
        self.epsilon = self.epsilon_init*(self.epsilon_decay**self.total_steps)


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
        central = self.greedysnake.SIZE // 2
        maxindex = action_map.argmax()
        row = maxindex // self.greedysnake.SIZE
        col = maxindex % self.greedysnake.SIZE - 1
        x = col - central
        y = central - row
        action = None
        if x >= 0 and y >= 0:
            if abs(y / x) < 1 or x == 0:
                action = Direction.RIGHT
            else:
                action = Direction.UP
        if x < 0 and y >= 0:
            if abs(y / x) < 1:
                action = Direction.UP
            else:
                action = Direction.LEFT
        if x < 0 and y < 0:
            if abs(y / x) < 1:
                action = Direction.LEFT
            else:
                action = Direction.DOWN
        if x >= 0 and y < 0:
            if abs(y / x) < 1 or x == 0:
                action = Direction.RIGHT
            else:
                action = Direction.DOWN
        return action

        
    def get_adhdp(self):

        lr = self.critic_net_learnrate_init
        clipnorm = self.critic_net_clipnorm

        # critic layers
        critic_model = keras.Sequential([
            keras.layers.Input(shape = (self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size + 1)), 
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(1, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Flatten(),
            keras.layers.Dense(self.greedysnake.SIZE ** 2, activation = 'elu', kernel_initializer='glorot_normal'),
            keras.layers.Dense((self.greedysnake.SIZE ** 2) // 2, activation = 'elu', kernel_initializer='glorot_normal'),
            keras.layers.Dense(1, activation='tanh', kernel_initializer='glorot_normal'),
        ], name = 'critic')

        # actor layers
        actor_model = keras.Sequential([
            keras.layers.Input(shape = (self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size)), 
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(self.timeslip_size * 4, (3, 3), padding='same', activation='elu', kernel_initializer='glorot_normal'),
            keras.layers.Conv2D(1,(3, 3), activation='tanh', padding='same', kernel_initializer='glorot_normal'),
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
                    frame[row, col] = 0.2
                    display += 'O'

            # food
            elif (np.array([row, col]) == self.greedysnake.food).all():
                frame[row, col] = 1.0
                display += '#'
            
            # block
            else: 
                display += '-'

            # switch line
            if col == self.greedysnake.SIZE - 1:
                display += '\n'
            # store frame to timeslip

        self.timeslip = np.insert(self.timeslip, 0, frame, axis=2)
        self.timeslip = np.delete(self.timeslip, self.timeslip_size, axis=2)

        return display
        
    def drive(self):

        # define stdscr for linux
        #stdscr = curses.initscr()
        #curses.noecho()
        #curses.cbreak()

        # record random initial steps
        for i in range(self.timeslip_size + 1):
            ram = self.random_action_map()
            a = self.get_action(ram)
            self.greedysnake.step(a)
            display = self.write_to_timeslip()
            print('=========Initial Steps===========')
            print(display)
        
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

            # open file to record steps
            f = open(self.train_hist_file, 'a')
            
            for i in range(self.max_steps):

                # observe state and action at t = 0
                if i == 0:
                    s_t = self.timeslip

                    # eps greedy
                    rand = np.random.rand()
                    #if rand <= (1-self.epsilon):
                    actmap_t = adhdp.predict_actor(s_t.reshape(1, self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size))
                    #else:
                    #    actmap_t = self.random_action_map().reshape((1, self.greedysnake.SIZE, self.greedysnake.SIZE, 1))

                    a_t = self.get_action(np.array(actmap_t).reshape(self.greedysnake.SIZE, self.greedysnake.SIZE))
                else: 
                    s_t = s_t_temp
                    a_t = a_t_temp
                    actmap_t = actmap_t_temp
                s_a_t = tf.concat([s_t, np.array(actmap_t).reshape((self.greedysnake.SIZE, self.greedysnake.SIZE, 1))], axis=2)

                # DEBUG
                #print('============ s_a_t ===================')
                #print(s_a_t.shape)
                #print(s_a_t[:,:,0])
                #print(s_a_t[:,:,1])
                #print(s_a_t[:,:,2])
                #print(s_a_t[:,:,3])
                #print(s_a_t[:,:,4])
                #print(s_a_t[:,:,5])
                #print(s_a_t[:,:,6])
                #print(s_a_t[:,:,7])
                #print(s_a_t[:,:,8])
                #print(s_a_t[:,:,9])
                #print(s_a_t[:,:,10])
                #print(s_a_t[:,:,11])
                #print(s_a_t[:,:,12])
                #print('========== s_a_t ==================')

                s_arr.append(s_t)
                s_a_arr.append(s_a_t)

                # take action via eps greedy, get reward
                signal = self.greedysnake.step(a_t)
                r = None
                if signal == Signal.HIT:
                    r = -1
                    hits += 1
                    self.greedysnake.reset()
                elif signal == Signal.EAT:
                    r = 1
                    eats += 1
                elif signal == Signal.NORMAL:
                    r = 0
                r_arr.append(r)

                # observe state after action
                s_t = np.copy(self.timeslip) #backup s_t
                display = self.write_to_timeslip()
                s_t_add_1 = self.timeslip
                s_t_temp = s_t_add_1
                
                # choose action at t+1
                # eps greedy
                rand = np.random.rand()
                #actmap_t_add_1 = None
                #if rand <= (1-self.epsilon):
                actmap_t_add_1 = adhdp.predict_actor(np.array(s_t_add_1).reshape(1, self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size))
                
                #else:
                #    actmap_t_add_1 = self.random_action_map().reshape((1, self.greedysnake.SIZE, self.greedysnake.SIZE, 1))
                a_t_add_1 = self.get_action(np.array(actmap_t_add_1).reshape(self.greedysnake.SIZE, self.greedysnake.SIZE))
                actmap_t_temp = actmap_t_add_1
                #print('=============== actmap(temp) ======================')
                #print(actmap_t_temp)

                a_t_temp = a_t_add_1

                # get teacher for critic net (online learning)
                #s_a_t_add_1 = self.concatenate_timeslip_and_actionmap(s_t_add_1, actmap_t_add_1)
                s_a_t_add_1 = tf.concat([s_t_add_1, actmap_t_add_1[0,:,:,:]], axis=2)
                q_t = critic_model.predict(np.array(s_a_t).reshape(1, self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size + 1))
                q_t_add_1 = critic_model.predict(np.array(s_a_t_add_1).reshape(1, self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size + 1))
                t = r + self.gamma * q_t_add_1
                if signal == Signal.HIT:
                    t = r
                t_arr.append(t)

                # accumulate index
                self.total_steps += 1

                # update learn rate and eps
                self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
                self.actor_net_learnrate = self.actor_net_learnrate_init * (self.actor_net_learnrate_decay ** self.total_steps)
                self.epsilon = self.epsilon_init*(self.epsilon_decay**self.total_steps)
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
                print('Step = ' + str(i) + ' / Epoch = ' + str(e) + ' / Total Steps = ' + str(self.total_steps) + ' / epsilon = ' + str(self.epsilon))
                print('action = ' + a_print + ' / reward = ' + r_print)
                print('teacher(Q) = ' + t_print + ' / predict(Q) = ' + predict_print +' / diff = ' + diff_print)
                print('Thousand steps average score = ' + str(avg))
                print('Hit rate = ' + str(hits / self.total_steps))
                print('Eat rate = ' + str(eats / self.total_steps))
                print(display)

                # print for linux
                #stdscr.addstr(0, 0, 'Step = ' + str(i) + '\tEpoch = ' + str(e) + '\tTotal Steps = ' + str(self.total_steps))
                #stdscr.addstr(1, 0, 'action = ' + a_print)
                #stdscr.addstr(2, 0, 'reward = ' + r_print)
                #stdscr.addstr(3, 0, 'teacher(Q) = ' + t_print)
                #stdscr.addstr(4, 0, 'predict(Q) = ' + str(float(predict_print)))
                #stdscr.addstr(6, 0, 'critic net learn rate = ' + str(float(self.critic_net_learnrate)))
                #stdscr.addstr(7, 0, 'Score = ' + str(len(self.greedysnake.snake)))
                #stdscr.addstr(8, 0, 'Thousand steps average score = ' + str(avg))
                #stdscr.addstr(9, 0, 'Hit rate = ' + str(hits / self.total_steps))
                #stdscr.addstr(10, 0, 'Eat rate = ' + str(eats / self.total_steps))
                #stdscr.addstr(11, 0, display)
                #stdscr.refresh()
                
            # train steps
            s = np.array(s_arr, dtype=np.float32).reshape((len(s_arr), self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size))
            s_a = np.array(s_a_arr, dtype=np.float32).reshape((len(s_a_arr), self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size + 1))
            t = np.array(t_arr, dtype=np.float32).reshape((len(t_arr), 1))
            critic_hist = critic_model.fit(s_a, t, epochs=self.critic_net_epochs, verbose=1, batch_size = self.batch_size)
            actor_hist = adhdp.fit(s, t, epochs=self.actor_net_epochs, verbose=1, batch_size = self.batch_size)

            # record train history
            #f.write(str(critic_hist.history)+'\n')
            #f.write(str(actor_hist.history)+'\n')
            f.close()

            # save model to file
            critic_model.save(self.critic_model_file)
            #adhdp.save(self.actor_model_file) # BUG saving subclass model adhdp not succeed


if __name__ == "__main__":
    d = Driver()
    #try:
    d.drive()
    #except:
    #    curses.echo()
    #    curses.nocbreak()
    #    curses.endwin()
    #finally:
    #    curses.echo()
    #    curses.nocbreak()
    #    curses.endwin()
        
