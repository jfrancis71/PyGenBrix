import pygame


pygame.init()
pygame.joystick.init()


class Joystick():
    def __init__(self):
        self.py_joystick = pygame.joystick.Joystick(0)
        self.py_joystick.init()
        self.left = 2
        self.right = 1
        self.up = 3
        self.down = 0

    def act(self):
        pygame.event.get()
        for i in range(4):
            if self.py_joystick.get_button(i):
                return i

# Note these bindings are specific for pong
class HumanAgent():
    def __init__(self):
        self.joystick = Joystick()

    def act(self, observation, on_policy, demo=False):
        position = self.joystick.act()
        if position == self.joystick.up:
            action = 2
        elif position == self.joystick.down:
            action = 3 
        else:
            action = 0
        return action

    def observe(self, observation, reward, done, reset):
        pass
