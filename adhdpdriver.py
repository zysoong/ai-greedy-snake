from greedysnake import GreedySnake, Direction, Signal
import time
import numpy as np
import curses
from threading import Thread
import subprocess
import tensorflow as tf
from tensorflow import keras
from collections import OrderedDict
import random
import configparser
import warnings
warnings.filterwarnings("ignore")


class Driver:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('dqndriver.ini')
        self.env = config['env']
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.max_epochs = int(config[self.env]['max_epochs'])
        self.max_steps = int(config[self.env]['max_steps'])
        self.critic_net_epochs = int(config[self.env]['critic_net_epochs'])
        self.beta = float(config[self.env]['beta'])
        self.gamma = float(config[self.env]['gamma'])
        self.critic_net_learnrate_init = float(config[self.env]['critic_net_learnrate_init'])
        self.critic_net_learnrate_decay = float(config[self.env]['critic_net_learnrate_decay'])
        self.critic_net_clipnorm = float(config[self.env]['critic_net_clipnorm'])
        self.policy_net_learnrate_init = float(config[self.env]['policy_net_learnrate_init'])
        self.policy_net_learnrate_decay = float(config[self.env]['policy_net_learnrate_decay'])
        self.policy_net_clipnorm = float(config[self.env]['critic_net_clipnorm'])
        self.epsilon_init = float(config[self.env]['epsilon_init'])
        self.epsilon_decay = float(config[self.env]['epsilon_decay'])
        self.train_hist_file = config[self.env]['train_hist_file']
        self.critic_model_file = config[self.env]['critic_model_file']
        self.policy_model_file = config[self.env]['policy_model_file']
        self.timeslip_size = config[self.env]['timeslip_size']
        self.timeslip = np.zeros(shape=(self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size))

        # parameters
        self.total_steps = 0
        self.beta = self.beta_init * (self.beta_decay ** self.total_steps)
        self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
        self.policy_net_learnrate = self.policy_net_learnrate_init * (self.policy_net_learnrate_decay ** self.total_steps)


    def critic_net(self):

        lr = self.critic_net_learnrate
        clipnorm = self.critic_net_clipnorm

        # define deep learning network
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (5, 5)), padding='same', activation='relu', input_shape=(self.timeslip.shape))
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(1, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(self.greedysnake.SIZE ** 2, kernel_initializer='he_normal', activation = 'relu'))
        model.add(keras.layers.Dense((self.greedysnake.SIZE ** 2) // 2, kernel_initializer='he_normal', activation = 'relu'))
        model.add(keras.layers.Dense(1, activation = 'tanh'))
        opt = keras.optimizers.RMSprop(
            lr = lr, 
            clipnorm = clipnorm
        )
        model.compile(loss = 'mean_squared_error', optimizer = opt, metrics=['MeanSquaredError'])
        return model

    def policy_net(self, critic_net):

        lr = self.policy_net_learnrate
        clipnorm = self.policy_net_clipnorm
        input = keras.layers.Input()

        # define deep learning network
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (5, 5)), padding='same', activation='relu', input_shape=(self.timeslip.shape))
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(self.timeslip_size * 4, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Conv2D(1, (3, 3)), padding='same', activation='relu')
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(self.greedysnake.SIZE ** 2, kernel_initializer='he_normal', activation = 'relu'))
        model.add(keras.layers.Dense((self.greedysnake.SIZE ** 2) // 2, kernel_initializer='he_normal', activation = 'relu'))
        model.add(keras.layers.Dense(4, activation = 'softmax'))

        # define error function
        y = critic_net.predict(input)
        def err_func(y_true, y_pred):
            t = tf.constant(1.0)
            y_t = tf.math.subtract(y, t)
            y_t_2 = tf.math.square(y_t)
            return tf.math.multiply(tf.constant(0.5), y_t_2)
        opt = keras.optimizers.RMSprop(
            lr = lr, 
            clipnorm = clipnorm
        )
        model.compile(loss = err_func, optimizer = opt, metrics=['MeanSquaredError'])
        return model

    def get_action(self, policy_net):
        act_arr = []
        output = np.array(policy_net.predict(self.timeslip))
        up = np.array([1, 0, 0, 0])
        down = np.array([0, 1, 0, 0])
        left = np.array([0, 0, 1, 0])
        right = np.array([0, 0, 0, 1])
        dist_up = np.linalg.norm(output - up)
        dist_down = np.linalg.norm(output - down)
        dist_left = np.linalg.norm(output - left)
        dist_right = np.linalg.norm(output - right)
        dict = {Direction.UP: float(dist_up), Direction.DOWN: float(dist_down), 
                Direction.LEFT: float(dist_left), Direction.RIGHT: float(dist_right)}
        sorted_dict = {k: v for k, v in sorted(dict.items(), reverse=False, key=lambda item: item[1])}
        for key in sorted_dict:
            act_arr.append(key)
        return act_arr[0]

    def write_to_timeslip(self):
        
        display = ''
        frame = np.zeros(shape=(self.greedysnake.SIZE, self.greedysnake.SIZE))

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
            self.timeslip = np.delete(self.timeslip, self.timeslip.shape[2]-1, axis=2)

        return display


    def drive(self):

        # define stdscr for linux
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        
        # define deep learning network
        critic_net = self.critic_net()
        policy_net = self.policy_net(critic_net)

        # initial: fullfill timeslip
        for i in range(self.timeslip_size):
            rand = random.randint(0, 3)
            action = None
            if rand == 0:
                action = Direction.UP
            elif rand == 1:
                action = Direction.DOWN
            elif rand == 2:
                action = Direction.LEFT
            elif rand == 3:
                action = Direction.RIGHT
            self.greedysnake.step(action)
            self.write_to_timeslip()

        # statics
        scores = []
        hits = 0
        eats = 0
        
        # on-policy ADHDP Learning
        for e in range(self.max_epochs):

            # execute steps for greedy snake
            s_arr = []
            a_arr = []
            r_arr = []
            t_arr = []

            # index to compare to max_steps
            i = 0

            # buffer
            s_t_temp = None
            a_t_temp = None

            # open file to record steps
            f = open(self.train_hist_file, 'a')

            # survival steps
            survival_steps = 0
            
            while i < self.max_steps:

                # observe state and action at t = 0
                if i == 0:
                    s_t = np.copy(self.timeslip)
                    a_t = self.get_action(policy_net)
                else: 
                    s_t = s_t_temp
                    a_t = a_t_temp

                # take action via greedy, get reward
                signal = self.greedysnake.step(a_t)
                r = None
                if signal == Signal.HIT:
                    survival_steps = 0
                    r = - (self.greedysnake.SIZE ** 2)
                    hits += 1
                    self.greedysnake.reset()
                elif signal == Signal.EAT:
                    survival_steps += 1
                    r = len(self.greedysnake.snake)
                    eats += 1
                elif signal == Signal.NORMAL:
                    survival_steps += 1
                    r = (1 - survival_steps / len(self.greedysnake.snake)) * len(self.greedysnake.snake)
                    if r < 0.1:
                        r = 0.1

                # observe state after action
                display = self.write_to_timeslip()
                s_t_add_1 = self.timeslip
                s_t_temp = s_t_add_1
                
                # choose action at t+1
                a_t_add_1 = self.get_action(policy_net)
                a_t_temp = a_t_add_1


                # get teacher
                v_s = critic_net.predict(s_t)
                v_s_t_add_1 = critic_net.predict(s_t_add_1)
                t = r + self.gamma *  v_s_t_add_1

                # store to tuple
                s_arr.append(s_t)
                a_arr.append(a_t)
                r_arr.append(r)
                t_arr.append(t)

                # accumulate index
                i += 1
                self.total_steps += 1

                # update learn rate
                self.beta = self.beta_init * (self.beta_decay ** self.total_steps)
                self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
                self.policy_net_learnrate = self.policy_net_learnrate_init * (self.policy_net_learnrate_decay ** self.total_steps)
                K.set_value(critic_net.optimizer.learning_rate, self.critic_net_learnrate)
                K.set_value(policy_net.optimizer.learning_rate, self.policy_net_learnrate)

                # store step info to file to retrain
                a_print = str(a_t)
                r_print = str(float(r))
                t_print = str(float(t))

                # calc stats
                if len(scores) < 1000:
                    scores.append(len(self.greedysnake.snake))
                else:
                    scores.pop(0)
                    scores.append(len(self.greedysnake.snake))
                avg = sum(scores) / len(scores)

                # print to debug
                #print('Step = ' + str(i) + ' / Epoch = ' + str(e) + ' / Total Steps = ' + str(self.total_steps))
                #print('action = ' + a_print + ' / reward = ' + r_print + ' / teacher = ' + t_print + '\n')
                #print(display)

                # print for linux
                stdscr.addstr(0, 0, 'Step = ' + str(i) + '\tEpoch = ' + str(e) + '\tTotal Steps = ' + str(self.total_steps))
                stdscr.addstr(1, 0, 'action = ' + a_print)
                stdscr.addstr(2, 0, 'reward = ' + r_print)
                stdscr.addstr(3, 0, 'teacher = ' + t_print)
                stdscr.addstr(4, 0, 'predict = ' + str(float(n_s_a)))
                stdscr.addstr(5, 0, 'predict / teacher diff rate = ' + str(abs(float(t) - float(n_s_a) / float(t))))
                stdscr.addstr(6, 0, 'beta = ' + str(float(self.beta)))
                stdscr.addstr(7, 0, 'critic net learn rate = ' + str(float(self.critic_net_learnrate)))
                stdscr.addstr(8, 0, 'Score = ' + str(len(self.greedysnake.snake)))
                stdscr.addstr(9, 0, 'Thousand steps average score = ' + str(avg))
                stdscr.addstr(10, 0, 'Hit rate = ' + str(hits / self.total_steps))
                stdscr.addstr(11, 0, 'Eat rate = ' + str(eats / self.total_steps))
                stdscr.addstr(12, 0, display)
                stdscr.refresh()
                
            # train critic net
            input = np.array(s_arr).reshape((len(s_arr), self.greedysnake.SIZE, self.greedysnake.SIZE, self.timeslip_size))
            teacher = np.array(t_arr).reshape((len(t_arr), 1))
            hist_critic = critic_net.fit(input, teacher, epochs=self.critic_net_epochs, batch_size = int(self.max_steps / 10), verbose=0)

            # train policy net. teacher could be any
            teacher_policy = np.zeros(shape=(len(t_arr), 4))
            hist_policy = critic_net.fit(input, teacher_policy, epochs=self.policyc_net_epochs, batch_size = int(self.max_steps / 10), verbose=0)

            # record train history
            f.write(str(hist_critic.history)+'\n')
            f.write(str(hist_policy.history)+'\n')
            f.close()

            # save model to file
            critic_net.save(self.critic_model_file)
            policy_net.save(self.policy_model_file)


if __name__ == "__main__":
    d = Driver()
    try:
        d.drive()
    except:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
        