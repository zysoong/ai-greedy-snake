from greedysnake import GreedySnake, Direction, Signal
import time
import numpy as np
from threading import Thread
import subprocess
from tensorflow import keras
from collections import OrderedDict

class Driver:

    def __init__(self, max_epochs = 1000000000000, max_steps = 8000, 
                max_teaching_epochs = 10, beta = 0.2, gamma = 0.5, beta_rate = 0.999, gamma_rate = 0.999):
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.max_teaching_epochs = max_teaching_epochs
        self.beta = beta
        self.gamma = gamma


    def display_game(self):
        display = ''
        for i in range(self.greedysnake.SIZE): 
            for j in range(self.greedysnake.SIZE):
                if (self.greedysnake.food == np.array([i, j])).all():
                    display += '#'
                elif self.greedysnake.is_snake(i, j) >= 0:
                    display += '@'
                else:
                    display += '-'
            display += '\n'
        print(display)



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
        dict = {float(qvalue_a1): Direction.UP, float(qvalue_a2): Direction.DOWN, 
                float(qvalue_a3): Direction.LEFT, float(qvalue_a4): Direction.RIGHT}
        od = OrderedDict(sorted(dict.items()), reverse=True)
        for k, v in od.items():
            return k, v


    def choose_action_via_boltzmann(self, state_action_arr, model, current_step, T_init=10, T_decay=0.999):
        

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

        # Boltzmann algo
        denom_of_counter = T_init * (T_decay ** current_step)
        if denom_of_counter < 0.1:
            denom_of_counter = 0.1
        counter_a1 = np.exp(qvalue_a1 / denom_of_counter)
        counter_a2 = np.exp(qvalue_a2 / denom_of_counter)
        counter_a3 = np.exp(qvalue_a3 / denom_of_counter)
        counter_a4 = np.exp(qvalue_a4 / denom_of_counter)
        denom = counter_a1 + counter_a2 + counter_a3 + counter_a4
        p_a1 = float(counter_a1 / denom)
        p_a2 = float(counter_a2 / denom)
        p_a3 = float(counter_a3 / denom)
        p_a3 = float(counter_a4 / denom)

        rand = np.random.randint(0, 1000)
        if 0 <= rand < 1000*p_a1:
            action = Direction.UP
            qvalue = qvalue_a1
        elif 1000*(p_a1) <= rand < 1000*(p_a1 + p_a2):
            action = Direction.DOWN
            qvalue = qvalue_a2
        elif 1000*(p_a1 + p_a2) <= rand < 1000*(p_a1 + p_a2 + p_a3):
            action = Direction.LEFT
            qvalue = qvalue_a3
        elif rand >= 1000*(p_a1 + p_a2 + p_a3):
            action = Direction.RIGHT
            qvalue = qvalue_a4
        else:

            # Small number prevension: Use greedy
            dict = {float(qvalue_a1): Direction.UP, float(qvalue_a2): Direction.DOWN, 
                float(qvalue_a3): Direction.LEFT, float(qvalue_a4): Direction.RIGHT}
            od = OrderedDict(sorted(dict.items()), reverse=True)
            for k, v in od.items():
                return k, v

        return qvalue, action


    def drive(self):

        # define deep learning network
        state_action_arr = self.convert_to_state_action_arr()[0]
        state_action_arr_dim = len(state_action_arr)
        print('state action array dimentions = ' + str(state_action_arr_dim))
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(20, input_dim = state_action_arr_dim, kernel_initializer='random_normal', activation = 'relu'))
        #model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(10, kernel_initializer='random_normal', activation = 'relu'))
        #model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(10, kernel_initializer='random_normal', activation = 'relu'))
        #model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(10, kernel_initializer='random_normal', activation = 'relu'))
        #model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(10, kernel_initializer='random_normal', activation = 'relu'))
        #model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(1))
        model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.RMSprop(lr=0.2), metrics=['MeanSquaredError'])
        
        # pretrain network with previous steps
        #from greedysnake import Direction
        #f = open('step.input', 'r')
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
        

        # set global step counter
        total_steps = 0

        
        
        # off-policy Q-Learning
        for e in range(self.max_epochs):

            # execute steps for greedy snake
            sat_arr = []
            r_arr = []
            satadd1_arr = []
            t_arr = []

            # index to compare to max_steps
            i = 0

            # total survival steps
            survival_steps = 0
            
            # buffer
            s_t_temp = None

            # open file to record steps
            f = open('step.input', 'a')
            
            while i < self.max_steps:

                # observe state and action at t = 0
                if i == 0:
                    s_t = self.convert_to_state_action_arr()[0]
                    a_t = self.choose_action_via_boltzmann(s_t, model, total_steps)[1]
                else: 
                    s_t = s_t_temp
                    a_t = self.choose_action_via_boltzmann(s_t, model, total_steps)[1]
                    
                # combine state and action at t
                s_a_t = self.combine_state_action_arr(s_t, a_t)

                # take action via greedy, get reward
                signal = self.greedysnake.step(a_t)
                r = 0
                if signal == Signal.HIT:
                    r = 0
                    survival_steps = 0
                    print('Game Over')
                    self.greedysnake.reset()
                elif signal == Signal.EAT:
                    survival_steps += 1
                    r = len(self.greedysnake.snake)
                elif signal == Signal.NORMAL:
                    survival_steps += 1

                    # avoid beginning reward
                    if len(self.greedysnake.snake) == 2:
                        r = 1
                    else:
                        r = len(self.greedysnake.snake) - (len(self.greedysnake.snake) * ( survival_steps / (self.greedysnake.SIZE * 2)))

                    # limitation
                    if r < 0.01:
                        r = 0.01

                # observe state after action
                s_t_add_1, display = self.convert_to_state_action_arr()
                s_t_temp = s_t_add_1

                # show step
                print('Step = ' + str(i) + ' / Epoch = ' + str(e) + ' / Total Steps = ' + str(total_steps))
                print(display)

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

                # store step info to file to track and pretrain
                s_print = str(list(s_t))
                a_print = str(a_t)
                r_print = str(float(r))
                sat_print = str(list(s_a_t))
                t_print = str(float(t))
                f.write('[' + s_print + ',' + a_print + ',' + r_print + ',' + sat_print + ',' + t_print +']\n')
                

            # record steps
            f.close()

            # train N(s, a) network
            input = np.array(sat_arr).reshape((len(sat_arr), state_action_arr_dim))
            teacher = np.array(t_arr).reshape((len(t_arr), 1))
            model.fit(input, teacher, epochs=self.max_teaching_epochs, batch_size = int(self.max_steps / 10))



    def run(self):
        self.drive()

    
d = Driver()
d.run()