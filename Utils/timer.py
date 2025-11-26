import time
import numpy as np

class Timer:
    def __init__(self):
        self.times = []

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

    def get_epochtime(self):
        days = self.times[-1] // 86400
        hours = self.times[-1] % 86400 // 3600
        minutes = self.times[-1] % 86400 % 3600 // 60
        seconds = self.times[-1] % 86400 % 3600 % 60
        if days == 0:
            if hours == 0:
                if minutes == 0:
                    return f'{seconds:.0f}s'
                return f'{minutes:.0f}m{seconds:.0f}s'
            return f'{hours:.0f}h{minutes:.0f}m{seconds:.0f}s'
        return f'{days:.0f}d{hours:.0f}h{minutes:.0f}m{seconds:.0f}s'

    def get_sumtime(self):
        days = self.sum() // 86400
        hours = self.sum() % 86400 // 3600
        minutes = self.sum() % 86400 % 3600 // 60
        seconds = self.sum() % 86400 % 3600 % 60
        if days == 0:
            if hours == 0:
                if minutes == 0:
                    return f'{seconds:.0f}s'
                return f'{minutes:.0f}m{seconds:.0f}s'
            return f'{hours:.0f}h{minutes:.0f}m{seconds:.0f}s'
        return f'{days:.0f}d{hours:.0f}h{minutes:.0f}m{seconds:.0f}s'

