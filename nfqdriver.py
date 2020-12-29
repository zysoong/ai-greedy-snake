from greedysnake import GreedySnake, Direction, Signal
import time
import numpy as np
import curses
from threading import Thread
import subprocess
from tensorflow import keras
from collections import OrderedDict
import random

class Driver:

    def __init__(self, max_epochs = 1000, max_steps = 6000, 
                max_teaching_epochs = 20, beta = 0.01, gamma = 0.3):
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.max_teaching_epochs = max_teaching_epochs
        self.beta = beta
        self.gamma = gamma


    def convert_to_state_action_arr(self):
        
        state_action_arr = np.zeros(self.greedysnake.SIZE * self.greedysnake.SIZE + 4)
        display = ''

        # generate states for N(s, a)
        for i in range(len(state_action_arr) - 4):
            row = i // self.greedysnake.SIZE
            col = i % self.greedysnake.SIZE
            snake_index = self.greedysnake.is_snake(row, col)

            # snake
            if snake_index > -1:

                # snake head
                if snake_index == 0: 
                    state_action_arr[i] = 0.5
                    display += '@'

                # snake body
                else:
                    state_action_arr[i] = 0.2
                    display += 'O'

            # food
            elif (np.array([row, col]) == self.greedysnake.food).all():
                state_action_arr[i] = 1.0
                display += '#'
            
            # block
            else: 
                display += '-'

            # switch line
            if col == self.greedysnake.SIZE - 1:
                display += '\n'

        return state_action_arr, display
    
    '''
    def convert_to_state_action_arr(self):
        
        state_action_arr = np.zeros(self.greedysnake.SIZE * self.greedysnake.SIZE * 4 + 4)
        display = ''

        for i in range(self.greedysnake.SIZE):
            for j in range(self.greedysnake.SIZE):

                snake_index = self.greedysnake.is_snake(i, j)

                # snake
                if snake_index > -1:

                    # snake head
                    if snake_index == 0: 
                        state_action_arr[i * self.greedysnake.SIZE + j * 4 + 0] = 1
                        display += '@'

                    # snake body
                    else:
                        state_action_arr[i * self.greedysnake.SIZE + j * 4 + 1] = 1
                        display += 'O'

                # food
                elif (np.array([i, j]) == self.greedysnake.food).all():
                    state_action_arr[i * self.greedysnake.SIZE + j * 4 + 2] = 1
                    display += '#'

                # block
                else: 
                    state_action_arr[i * self.greedysnake.SIZE + j * 4 + 3] = 1
                    display += '-'

                # switch line
                if j == self.greedysnake.SIZE - 1:
                    display += '\n'

        return state_action_arr, display
    '''


    def combine_state_action_arr(self, state_action_arr, action): 
        result = state_action_arr.copy()
        if action == Direction.UP:
            result[len(state_action_arr) - 4] = 1
            result[len(state_action_arr) - 3] = 0
            result[len(state_action_arr) - 2] = 0
            result[len(state_action_arr) - 1] = 0
        elif action == Direction.DOWN:
            result[len(state_action_arr) - 4] = 0
            result[len(state_action_arr) - 3] = 1
            result[len(state_action_arr) - 2] = 0
            result[len(state_action_arr) - 1] = 0
        elif action == Direction.LEFT:
            result[len(state_action_arr) - 4] = 0
            result[len(state_action_arr) - 3] = 0
            result[len(state_action_arr) - 2] = 1
            result[len(state_action_arr) - 1] = 0
        elif action == Direction.RIGHT:
            result[len(state_action_arr) - 4] = 0
            result[len(state_action_arr) - 3] = 0
            result[len(state_action_arr) - 2] = 0
            result[len(state_action_arr) - 1] = 1
        return result
                
        
    def choose_action_via_greedy(self, state_action_arr, model):

        # get q values for all actions and compare the max q value
        # action 1 (UP)
        state_action_arr[len(state_action_arr) - 4] = 1
        state_action_arr[len(state_action_arr) - 3] = 0
        state_action_arr[len(state_action_arr) - 2] = 0
        state_action_arr[len(state_action_arr) - 1] = 0
        qvalue_a1 = model.predict(state_action_arr.reshape((1, len(state_action_arr))))


        # action 2 (DOWN)
        state_action_arr[len(state_action_arr) - 4] = 0
        state_action_arr[len(state_action_arr) - 3] = 1
        state_action_arr[len(state_action_arr) - 2] = 0
        state_action_arr[len(state_action_arr) - 1] = 0
        qvalue_a2 = model.predict(state_action_arr.reshape((1, len(state_action_arr))))

        # action 3 (LEFT)
        state_action_arr[len(state_action_arr) - 4] = 0
        state_action_arr[len(state_action_arr) - 3] = 0
        state_action_arr[len(state_action_arr) - 2] = 1
        state_action_arr[len(state_action_arr) - 1] = 0
        qvalue_a3 = model.predict(state_action_arr.reshape((1, len(state_action_arr))))

        # action 4 (RIGHT)
        state_action_arr[len(state_action_arr) - 4] = 0
        state_action_arr[len(state_action_arr) - 3] = 0
        state_action_arr[len(state_action_arr) - 2] = 0
        state_action_arr[len(state_action_arr) - 1] = 1
        qvalue_a4 = model.predict(state_action_arr.reshape((1, len(state_action_arr))))

        # figure out the best legal action
        q_arr = []
        act_arr = []
        dict = {Direction.UP: float(qvalue_a1), Direction.DOWN: float(qvalue_a2), 
                Direction.LEFT: float(qvalue_a3), Direction.RIGHT: float(qvalue_a4)}
        sorted_dict = {k: v for k, v in sorted(dict.items(), reverse=True, key=lambda item: item[1])}
        for key in sorted_dict:
            q_arr.append(sorted_dict[key])
            act_arr.append(key)
        return q_arr[0], act_arr[0]


    def choose_action_via_eps_greedy(self, eps, state_action_arr, model):

        # get q values for all actions and compare the max q value
        # action 1 (UP)
        state_action_arr[len(state_action_arr) - 4] = 1
        state_action_arr[len(state_action_arr) - 3] = 0
        state_action_arr[len(state_action_arr) - 2] = 0
        state_action_arr[len(state_action_arr) - 1] = 0
        qvalue_a1 = model.predict(state_action_arr.reshape((1, len(state_action_arr))))


        # action 2 (DOWN)
        state_action_arr[len(state_action_arr) - 4] = 0
        state_action_arr[len(state_action_arr) - 3] = 1
        state_action_arr[len(state_action_arr) - 2] = 0
        state_action_arr[len(state_action_arr) - 1] = 0
        qvalue_a2 = model.predict(state_action_arr.reshape((1, len(state_action_arr))))

        # action 3 (LEFT)
        state_action_arr[len(state_action_arr) - 4] = 0
        state_action_arr[len(state_action_arr) - 3] = 0
        state_action_arr[len(state_action_arr) - 2] = 1
        state_action_arr[len(state_action_arr) - 1] = 0
        qvalue_a3 = model.predict(state_action_arr.reshape((1, len(state_action_arr))))

        # action 4 (RIGHT)
        state_action_arr[len(state_action_arr) - 4] = 0
        state_action_arr[len(state_action_arr) - 3] = 0
        state_action_arr[len(state_action_arr) - 2] = 0
        state_action_arr[len(state_action_arr) - 1] = 1
        qvalue_a4 = model.predict(state_action_arr.reshape((1, len(state_action_arr))))

        # figure out the best legal action
        q_arr = []
        act_arr = []
        dict = {Direction.UP: float(qvalue_a1), Direction.DOWN: float(qvalue_a2), 
                Direction.LEFT: float(qvalue_a3), Direction.RIGHT: float(qvalue_a4)}
        sorted_dict = {k: v for k, v in sorted(dict.items(), reverse=True, key=lambda item: item[1])}
        for key in sorted_dict:
            q_arr.append(sorted_dict[key])
            act_arr.append(key)

        # eps-greedy
        rand = np.random.rand()
        if rand <= (1-eps):
            return q_arr[0], act_arr[0]
        else:
            index = random.randint(0, len(q_arr)-1)
            return q_arr[index], act_arr[index]

    def drive(self, stdscr):
        # define deep learning network
        state_action_arr = self.convert_to_state_action_arr()[0]
        state_action_arr_dim = len(state_action_arr)
        print('state action array dimentions = ' + str(state_action_arr_dim))
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(15, input_dim = state_action_arr_dim, kernel_initializer='he_normal', activation = 'elu'))
        model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(15, kernel_initializer='he_normal', activation = 'elu'))
        model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1))
        opt = keras.optimizers.RMSprop(
            lr = 0.01, 
            clipnorm=40
        )
        model.compile(loss = 'mean_squared_error', optimizer = opt, metrics=['MeanSquaredError'])
        
        # pretrain network with previous steps
        #from greedysnake import Direction
        #f = open('train.hist', 'r')
        #lines = f.readlines()
        #steps = [None]*len(lines)
        #sat_arr = []
        #r_arr = []
        #satadd1_arr = []
        #t_arr = []
        #for i in range(len(steps)):
        #    exec('steps[' + str(i) + '] = ' + str(lines[i]))
        #    sat_arr.append(self.combine_state_action_arr(steps[i][0], steps[i][1]))
        #    r_arr.append(steps[i][2])
        #    sat_arr.append(self.combine_state_action_arr(steps[i][0], steps[i][1]))


        # set step counter
        total_steps = 0

        # statics
        scores = []
        hits = 0
        eats = 0
        
        # off-policy Q-Learning
        for e in range(self.max_epochs):

            # execute steps for greedy snake
            sat_arr = []
            r_arr = []
            satadd1_arr = []
            t_arr = []

            # index to compare to max_steps
            i = 0

            # buffer
            s_t_temp = None

            # open file to record steps
            f = open('train_hist.log', 'a')
            
            while i < self.max_steps:

                # observe state and action at t = 0
                if i == 0:
                    s_t = self.convert_to_state_action_arr()[0]
                    a_t = self.choose_action_via_eps_greedy(1.0*(0.99997**total_steps), s_t, model)[1]
                else: 
                    s_t = s_t_temp
                    a_t = self.choose_action_via_eps_greedy(1.0*(0.99997**total_steps), s_t, model)[1]
                    
                # combine state and action at t
                s_a_t = self.combine_state_action_arr(s_t, a_t)

                # take action via greedy, get reward
                signal = self.greedysnake.step(a_t)
                r = None
                if signal == Signal.HIT:
                    r = - (self.greedysnake.SIZE ** 2)
                    hits += 1
                    self.greedysnake.reset()
                elif signal == Signal.EAT:
                    r = len(self.greedysnake.snake)
                    eats += 1
                elif signal == Signal.NORMAL:
                    r = 0

                # observe state after action
                s_t_add_1, display = self.convert_to_state_action_arr()
                s_t_temp = s_t_add_1
                
                # choose action at t+1
                max_q_t_add_1, a_t_add_1 = self.choose_action_via_greedy(s_t_add_1, model)

                # combine state and action at t+1
                s_a_t_add_1 = self.combine_state_action_arr(s_t_add_1, a_t_add_1)

                # get teacher
                n_s_a = model.predict(s_a_t.reshape((1, state_action_arr_dim)))
                t = n_s_a + self.beta * (r + self.gamma * max_q_t_add_1 - n_s_a)

                # store to tuple
                sat_arr.append(s_a_t)
                r_arr.append(r)
                satadd1_arr.append(s_a_t_add_1)
                t_arr.append(t)

                # accumulate index
                i += 1
                total_steps += 1

                # store step info to file to retrain
                a_print = str(a_t)
                r_print = str(float(r))
                sat_print = str(list(s_a_t))
                t_print = str(float(t))
                #f.write('[' + sat_print + ',' + t_print +']\n')

                # calc stats
                if len(scores) < 1000:
                    scores.append(len(self.greedysnake.snake))
                else:
                    scores.pop(0)
                    scores.append(len(self.greedysnake.snake))
                avg = sum(scores) / len(scores)

                # print to debug
                #print('Step = ' + str(i) + ' / Epoch = ' + str(e) + ' / Total Steps = ' + str(total_steps))
                #print('action = ' + a_print + ' / reward = ' + r_print + ' / teacher = ' + t_print + '\n')
                #print(display)

                # print for linux
                stdscr.addstr(0, 0, 'Step = ' + str(i) + '\tEpoch = ' + str(e) + '\tTotal Steps = ' + str(total_steps))
                stdscr.addstr(1, 0, 'action = ' + a_print)
                stdscr.addstr(2, 0, 'reward = ' + r_print)
                stdscr.addstr(3, 0, 'teacher = ' + t_print)
                stdscr.addstr(4, 0, 'predict = ' + str(float(n_s_a)))
                stdscr.addstr(5, 0, 'Score = ' + str(len(self.greedysnake.snake)))
                stdscr.addstr(6, 0, 'Thousand steps average score = ' + str(avg))
                stdscr.addstr(7, 0, 'Hit rate = ' + str(hits / total_steps))
                stdscr.addstr(8, 0, 'Eat rate = ' + str(eats / total_steps))
                stdscr.addstr(9, 0, display)
                stdscr.refresh()
                
            # train N(s, a) network
            input = np.array(sat_arr).reshape((len(sat_arr), state_action_arr_dim))
            teacher = np.array(t_arr).reshape((len(t_arr), 1))
            hist = model.fit(input, teacher, epochs=self.max_teaching_epochs, batch_size = int(self.max_steps / 10), verbose=0)

            # record train history
            f.write(str(hist))
            f.close()

            model.save('model')
            time.sleep(5)



    def run(self, stdscr):
        self.drive(stdscr)

stdscr = curses.initscr()
curses.noecho()
curses.cbreak()
try:
    d = Driver()
    d.run(stdscr)
finally:
    curses.echo()
    curses.nocbreak()
    curses.endwin()