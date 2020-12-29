from greedysnake import GreedySnake, Direction, Signal
import time
import numpy as np
from threading import Thread
import subprocess
from pynput.keyboard import Key, Listener
import os

class Driver:

    def __init__(self, flush_time = 0.5, display_delay = 0.0):
        self.greedysnake = GreedySnake()
        self.signal_in = Direction.STRAIGHT
        self.flush_time = flush_time
        self.display_delay = display_delay

    def on_press(self, key):
        if key == Key.up:
            self.signal_in = Direction.UP
        elif key == Key.down:
            self.signal_in = Direction.DOWN
        elif key == Key.left:
            self.signal_in = Direction.LEFT
        elif key == Key.right:
            self.signal_in = Direction.RIGHT
        else:
            self.signal_in = Direction.STRAIGHT
            #time.sleep(self.keyboard_delay)


    def monitor_game(self):
        while True:
            display = ''
            for i in range(13): 
                for j in range(13):
                    if (self.greedysnake.food == np.array([i-1, j-1])).all():
                        display += '#'
                    elif self.greedysnake.is_snake(i-1, j-1):
                        display += '@'
                    elif i == 0 or i == 12: 
                        display += '*'
                    elif j == 0 or j == 12: 
                        display += '|'
                    else:
                        display += '-'
                display += '\n'
            print(display, end='\r', flush=True)
            time.sleep(self.display_delay)

    def drive(self):
        while True:
            signal = self.greedysnake.step(self.signal_in)
            if signal == Signal.HIT: 
                print('Game Over')
                self.greedysnake.reset()
            time.sleep(self.flush_time)


    def run(self):
        Thread(target=self.drive).start()
        Thread(target=self.monitor_game).start()
        with Listener(on_press = self.on_press) as listener:
            listener.join()

    
d = Driver()
d.run()