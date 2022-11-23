class MovingAverage():
    def __init__(self, gamma=.999):
        self.init = False
        self.running_mean = 0.0
        self.gamma = gamma

    def update(self, value):
        if self.init is False:
            self.init = True
            self.running_mean = value
        self.running_mean = self.gamma*self.running_mean + (1.0-self.gamma)*value

    def value(self):
        if self.init is False:
            raise "Error, not initialized"
        return self.running_mean
