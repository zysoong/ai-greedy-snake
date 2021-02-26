from greedysnake import GreedySnake, Direction, Signal
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import configparser
from collections import deque
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
            action_map = self.actor(state)
            state_action = tf.concat([state, action_map], 1)
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

    def save_models(self):
        self.critic.save('adhdp_critic')
        self.actor.save('adhdp_actor')

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

    def get_action(self, state, adhdp):
        actor_output = adhdp.predict_actor(np.array(state).reshape((1, 8)))
        actor_sm = np.array(tf.nn.softmax(actor_output)).reshape((4))
        rand = np.random.rand()
        action = None
        if 0 <= rand < actor_sm[0]:
            action = Direction.UP
        elif actor_sm[0] <= rand < actor_sm[0] + actor_sm[1]:
            action = Direction.DOWN
        elif actor_sm[0] + actor_sm[1] <= rand < actor_sm[0] + actor_sm[1] + actor_sm[2]:
            action = Direction.LEFT
        else:
            action = Direction.RIGHT
        return action, actor_output

    def get_action_index(self, action):
        if action == Direction.UP:
            return 0
        elif action == Direction.DOWN:
            return 1
        elif action == Direction.LEFT:
            return 2
        elif action == Direction.RIGHT:
            return 3
        
    def get_adhdp(self):

        # critic layers
        critic_model = keras.Sequential([
            keras.layers.Input(shape = (12)), 
            keras.layers.Dense(32, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.Dense(15, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.Dense(1, activation='tanh', kernel_initializer='glorot_normal')
        ], name = 'critic')

        # critic layers
        actor_model = keras.Sequential([
            keras.layers.Input(shape = (8)), 
            keras.layers.Dense(32, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.Dense(15, activation = 'relu', kernel_initializer='glorot_normal'),
            keras.layers.Dense(4, kernel_initializer='glorot_normal')
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
        
        # critic model
        critic_model.compile(loss = keras.losses.MSE, optimizer = c_opt)

        # actor model
        adhdp = ADHDP(critic=critic_model, actor=actor_model)
        adhdp.compile(loss = keras.losses.MSE, optimizer = a_opt) # loss is MSE to compare the Q values
        return critic_model, adhdp


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
            get_action_result_current_temp = None
            action_current = None
            
            # start steps
            stamina = 0
            stamina_max = self.greedysnake.SIZE
            for i in range(self.max_steps):

                # observe state and action at t = 0
                if i == 0:
                    s_current = self.get_state()[0].reshape((1, 8))
                    get_action_result_current = self.get_action(s_current, adhdp)
                    a_current = np.array(get_action_result_current[1]).reshape((1, 4))
                    action_current = get_action_result_current[0]
                else: 
                    s_current = s_current_temp
                    get_action_result_current = get_action_result_current_temp
                    a_current = np.array(get_action_result_current[1]).reshape((1, 4))
                    action_current = get_action_result_current[0]
                s_memory.append(s_current)
                s_a_current = tf.concat([s_current, a_current], axis=1)
                s_a_memory.append(s_a_current)

                # take action via eps greedy, get reward
                signal = self.greedysnake.step(action_current)
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
                display = self.get_state()[1]
                s_future = self.get_state()[0].reshape((1, 8))
                s_current_temp = s_future
                
                # choose action at t+1
                get_action_result_future = self.get_action(np.array(s_future).reshape((1, 8)), adhdp)
                a_future = get_action_result_future[1]
                get_action_result_current_temp = get_action_result_future

                # get teacher for critic net
                s_a_future = tf.concat([np.array(s_future).reshape(1, 8), np.array(a_future).reshape(1, 4)], axis=1)
                q_current = critic_model(np.array(s_a_current).reshape((1, 12)))
                q_future = critic_model(np.array(s_a_future).reshape((1, 12)))
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
                print(s_future.reshape((2, 4)))

                # record training history
                if self.total_steps % 100 == 0:
                    f = open(self.train_hist_file, 'a+')
                    f.write(str(avg)+'\n')
                    f.close()
                
                
            # train steps
            s = np.array(list(s_memory), dtype=np.float32).reshape((len(list(s_memory)), 8))
            s_a = np.array(list(s_a_memory), dtype=np.float32).reshape((len(list(s_a_memory)), 12))
            t = np.array(list(t_memory), dtype=np.float32).reshape((len(list(t_memory)), 1))
            critic_model.fit(s_a, t, epochs=self.critic_net_epochs, verbose=0, batch_size = self.batch_size)
            adhdp.fit(s, t, epochs=self.actor_net_epochs, verbose=0, batch_size = self.batch_size)

            # save model to file
            adhdp.save_models()

if __name__ == "__main__":
    d = Driver()
    d.run()
        
