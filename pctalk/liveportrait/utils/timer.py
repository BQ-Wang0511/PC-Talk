
"""
tools to measure elapsed time
"""

import time

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    def tic(self):
        # Wall-clock timing remains consistent across worker threads.
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        return self.diff

    def clear(self):
        self.start_time = 0.
        self.diff = 0.
