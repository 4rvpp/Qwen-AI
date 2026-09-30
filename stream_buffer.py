"""Bounded capture buffer with replaceable partial hypotheses."""
from collections import deque
import numpy as np

class StreamBuffer:
    def __init__(self, rate, pause=.8, interval=2., max_phrase=12., max_pending=45.):
        if rate not in range(8000, 96001):
            raise ValueError('Unsupported sample rate')
        self.rate, self.pause, self.interval = rate, pause, interval
        self.max_phrase, self.max_pending = max_phrase, max_pending
        self.parts, self.pending = [], deque()
        self.preroll = deque(maxlen=2)
        self.count = self.quiet = self.last_preview = self.phrase = 0
        self.stopped = False

    @property
    def backlog(self):
        return (sum(len(a) for _, a in self.pending) + self.count) / self.rate

    def add(self, samples):
        if self.stopped:
            return
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not len(samples) or len(samples) > self.rate or not np.isfinite(samples).all():
            raise ValueError('Invalid audio frame')
        if self.backlog + len(samples) / self.rate > self.max_pending:
            raise OverflowError('CPU cannot keep up. Capture stopped; waiting for pending text. Start again when finished.')
        samples = np.clip(samples, -1, 1)
        speech = float(np.sqrt(np.mean(samples ** 2))) >= .008
        if not self.parts and not speech:
            self.preroll.append(samples)
            return
        if not self.parts:
            self.parts.extend(self.preroll)
            self.count = sum(len(a) for a in self.parts)
            self.preroll.clear()
        self.parts.append(samples)
        self.count += len(samples)
        self.quiet = 0 if speech else self.quiet + len(samples)
        if self.quiet / self.rate >= self.pause or self.count / self.rate >= self.max_phrase:
            self.finish_phrase()

    def finish_phrase(self):
        if self.parts:
            self.pending.append((self.phrase, np.concatenate(self.parts)))
            self.phrase += 1
        self.parts = []
        self.count = self.quiet = self.last_preview = 0

    def stop(self):
        self.finish_phrase()
        self.stopped = True

    def next_job(self):
        if self.pending:
            phrase, samples = self.pending.popleft()
            return phrase, samples, True
        if self.count - self.last_preview >= self.rate * self.interval:
            self.last_preview = self.count
            return self.phrase, np.concatenate(self.parts), False
        return None
