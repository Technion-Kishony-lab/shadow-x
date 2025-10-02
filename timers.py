import time
from contextlib import contextmanager


LAVELS_TO_TIMES_AND_COUNTS = {}


@contextmanager
def timeit(label, vervose=False):
    t = time.time()
    yield
    elapsed = time.time() - t
    if label not in LAVELS_TO_TIMES_AND_COUNTS:
        LAVELS_TO_TIMES_AND_COUNTS[label] = [0., 0]
    LAVELS_TO_TIMES_AND_COUNTS[label][0] += elapsed
    LAVELS_TO_TIMES_AND_COUNTS[label][1] += 1
    if vervose:
        print(f"{label}: {elapsed * 1000:.1f} ms")


def print_time_and_reset(label):
    if label in LAVELS_TO_TIMES_AND_COUNTS:
        total_time, count = LAVELS_TO_TIMES_AND_COUNTS[label]
        print(f"{label} avg: {total_time / count * 1000:.1f} ms over {count} calls")
        LAVELS_TO_TIMES_AND_COUNTS[label] = [0, 0]


def reset_timers():
    LAVELS_TO_TIMES_AND_COUNTS.clear()


def print_all_timers():
    for label, (total_time, count) in LAVELS_TO_TIMES_AND_COUNTS.items():
        print(f"{label:50} avg: {total_time / count * 1000:.1f} ms over {count} calls")
    LAVELS_TO_TIMES_AND_COUNTS.clear()

