import time


class Timer:
    def __init__(self, name):
        self.name = name
        self.times = []

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        self.times.append(elapsed_time)

    def reset(self):
        self.times = []

    def get_total_counts(self):
        return len(self.times)

    def get_total_time(self):
        return sum(self.times)

    def get_avg_time(self):
        count = self.get_total_counts()
        return self.get_total_time() / count if count > 0 else 0

    def report(self, reset=False):
        total_time = self.get_total_time()
        count = self.get_total_counts()
        avg_time = self.get_avg_time()
        if reset:
            self.reset()
        return f"{self.name}: {total_time:.2f}s over {count} runs, avg: {avg_time*1000:.1f} ms"
