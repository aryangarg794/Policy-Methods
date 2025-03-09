from collections import deque

class RollingAverage:
    def __init__(self, window_size):
        self.window = deque(maxlen=window_size)
        self.averages = []

    def update(self, value):
        self.window.append(value)
        self.averages.append(self.get_average)

    @property
    def get_average(self):
        return sum(self.window) / len(self.window) if self.window else 0.0