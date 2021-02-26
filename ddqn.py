from greedysnake import GreedySnake, Direction, Signal
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import configparser
import sys
import warnings
from collections import deque
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)

class Driver:

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('ddqn.ini')
        self.env = config['ENV']['env']
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.max_epochs = int(config[self.env]['max_epochs'])
        self.max_steps = int(config[self.env]['max_steps'])
        self.batch_size = int(config[self.env]['batch_size'])
        self.memory_size = int(config[self.env]['memory_size'])
        self.mini_batch_size = int(config[self.env]['mini_batch_size'])
        self.critic_net_epochs = int(config[self.env]['critic_net_epochs'])
        self.gamma = float(config[self.env]['gamma'])
        self.epsilon_init = float(config[self.env]['epsilon_init'])
        self.epsilon_decay = float(config[self.env]['epsilon_decay'])
        self.critic_net_learnrate_init = float(config[self.env]['critic_net_learnrate_init'])
        self.critic_net_learnrate_decay = float(config[self.env]['critic_net_learnrate_decay'])
        self.critic_net_clipnorm = float(config[self.env]['critic_net_clipnorm'])
        self.target_update_freq = float(config[self.env]['target_update_freq'])
        self.train_hist_file = config[self.env]['train_hist_file']

        # parameters
        self.total_steps = 0
        self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
        self.epsilon = self.epsilon_init * (self.epsilon_decay ** self.total_steps)

    def get_action(self, state, critic_model, epsilon):

        rand_strategy = np.random.rand()
        # random action
        if 0 <= rand_strategy <= epsilon:
            q = critic_model.predict(np.array(state).reshape((1, 8)))
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
            q = critic_model.predict(np.array(state).reshape((1, 8)))
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
        
    def get_ddqn(self):

        # critic layers
        critic_model = keras.Sequential([
            keras.layers.Input(shape = (8)), 
            keras.layers.Dense(32, activation = 'relu', kernel_initializer='random_normal'),
            keras.layers.Dense(15, activation = 'relu', kernel_initializer='random_normal'),
            keras.layers.Dense(4, kernel_initializer='random_normal')
        ], name = 'critic')

        # optimizer
        c_opt = keras.optimizers.Adam(
            lr = self.critic_net_learnrate, 
            clipnorm = self.critic_net_clipnorm
        )
        
        # critic model
        critic_model.compile(loss = keras.losses.MSE, optimizer = c_opt)

        # target model
        target = keras.models.clone_model(critic_model)
        target.set_weights(critic_model.get_weights())
        return critic_model, target


    def get_state(self):
        display = ''
        state = np.zeros(shape=(8))
        head = self.greedysnake.snake[0]
        head_up = head + np.array([-1, 0])
        head_down = head + np.array([1, 0])
        head_left = head + np.array([0, -1])
        head_right = head + np.array([0, 1])
        
        if self.greedysnake.is_snake(head_up[0], head_up[1]) != -1 or head_up[0] < 0:
            state[0] = 1.
        if self.greedysnake.is_snake(head_down[0], head_down[1]) != -1 or head_down[0] >= self.greedysnake.SIZE:
            state[1] = 1.
        if self.greedysnake.is_snake(head_left[0], head_left[1]) != -1 or head_left[1] < 0:
            state[2] = 1.
        if self.greedysnake.is_snake(head_right[0], head_right[1]) != -1 or head_right[1] >= self.greedysnake.SIZE:
            state[3] = 1.

        food_vec = self.greedysnake.food - head
        food_vec[0] = -food_vec[0]
        x = food_vec[1]
        y = food_vec[0]
        norm_max = np.sqrt(2 * (self.greedysnake.SIZE ** 2))
        norm = 1. - (np.linalg.norm(np.array(x, y)) / norm_max)

        if x == 0 and y >= 0:
            state[4] = norm
        elif x == 0 and y < 0:
            state[5] = norm
        elif x > 0 and y >= 0 and y / x <= 1:
            state[7] = norm
        elif x > 0 and y >= 0 and y / x > 1:
            state[4] = norm
        elif x < 0 and y >= 0 and y / x < -1:
            state[4] = norm
        elif x < 0 and y >= 0 and y / x >= -1:
            state[6] = norm
        elif x < 0 and y <= 0 and y / x <= 1:
            state[6] = norm
        elif x < 0 and y <= 0 and y / x > 1:
            state[5] = norm
        elif x > 0 and y <= 0 and y / x < -1:
            state[5] = norm
        elif x > 0 and y <= 0 and y / x >= -1:
            state[7] = norm
        

        # generate states for N(s, a)
        for i in range(self.greedysnake.SIZE ** 2):
            row = i // self.greedysnake.SIZE
            col = i % self.greedysnake.SIZE
            snake_index = self.greedysnake.is_snake(row, col)

            # snake
            if snake_index > -1:

                # snake head
                if snake_index == 0: 
                    display += '@'

                # snake body
                else:
                    display += 'O'

            # food
            elif (np.array([row, col]) == self.greedysnake.food).all():
                display += '#'
            
            # block
            else: 
                display += '-'

            # switch line
            if col == self.greedysnake.SIZE - 1:
                display += '\n'


        return state, display
        
    def run(self):
        
        # define deep learning network
        critic_model, target = self.get_ddqn()
        
        # statics
        scores = deque(maxlen=1000)
        max_score = 0
        hits = 0
        eats = 0

        for e in range(self.max_epochs):

            # execute steps for greedy snake
            s_arr = deque()
            s_a_future_arr = deque()
            r_arr = deque()
            t_arr = deque()
            q_arr = deque()

            # buffer
            s_current_temp = None
            a_current_temp = None
            
            # start steps
            stamina = 0
            stamina_max = self.greedysnake.SIZE
            for i in range(self.max_steps):

                # observe state and action at t = 0
                if i == 0:
                    s_current = self.get_state()[0].reshape((1, 8))
                    a_current = self.get_action(s_current, critic_model, self.epsilon)[0]
                else: 
                    s_current = s_current_temp
                    a_current = a_current_temp
                s_arr.append(s_current)

                # take action via eps greedy, get reward
                signal = self.greedysnake.step(a_current)
                r = None

                # signal reward
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

                r_arr.append(r)

                # observe state after action
                display = self.get_state()[1]
                s_future = self.get_state()[0].reshape((1, 8))
                s_current_temp = s_future
                s_a_future_arr.append(s_future)
                
                # choose action at t+1
                gares = self.get_action(s_future, critic_model, self.epsilon)
                a_future = gares[0]
                a_current_temp = a_future

                # get teacher for critic net (online learning)
                q_current = critic_model.predict(s_current)
                target_sa = target.predict(s_future)
                t = [0,0,0,0]
                index = np.argmax(np.array(target_sa).reshape((4)))
                for j in range(len(t)):
                    if j == self.get_action_index(a_current):
                        t[j] = r + self.gamma * np.array(target_sa).reshape((4))[index]
                        if signal == Signal.HIT and j == self.get_action_index(a_current):
                            t[j] = r
                    else:
                        t[j] = np.array(q_current).reshape((4))[j]
                q_arr.append(q_current)
                t_arr.append(t)

                # accumulate index
                self.total_steps += 1

                # update learn rate and eps
                self.critic_net_learnrate = self.critic_net_learnrate_init * (self.critic_net_learnrate_decay ** self.total_steps)
                self.epsilon = self.epsilon_init * (self.epsilon_decay ** self.total_steps)
                K.set_value(critic_model.optimizer.learning_rate, self.critic_net_learnrate)

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
                print('thousand steps average score = ' + str(avg))
                print('max avg. score = ' + str(max_score))
                print('Hit rate = ' + str(hits / self.total_steps))
                print('Eat rate = ' + str(eats / self.total_steps))
                print(display)
                print(str(np.array(s_future).reshape((2, 4))))

                
                if self.total_steps % self.target_update_freq == 0 and self.total_steps != 0:
                    print('clone critic weights to target')
                    target.set_weights(critic_model.get_weights())

                if self.total_steps % 1000 == 0:
                    print('models saved')
                    critic_model.save('ddqn_critic')
                    target.save('ddqn_target')

                if self.total_steps % 100 == 0:
                    f = open(self.train_hist_file, 'a+')
                    f.write(str(avg)+'\n')
                    f.close()

            # train steps
            s = np.array(s_arr, dtype=np.float32).reshape((len(s_arr), 8))
            t = np.array(t_arr, dtype=np.float32).reshape((len(t_arr), 4))
            r = np.array(r_arr, dtype=np.float32).reshape((len(r_arr), 1))
            critic_model.fit(s, t, epochs=self.critic_net_epochs, verbose=0, batch_size = self.batch_size)

            # record train history
            #f.write(str(critic_hist.history)+'\n')
            #f.write(str(actor_hist.history)+'\n')
            #f.close()


if __name__ == "__main__":
    d = Driver()
    d.run()
        
