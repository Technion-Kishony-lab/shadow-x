import time
from contextlib import contextmanager


LABELS_TO_TIMES_AND_COUNTS = {}


@contextmanager
def timeit(label, vervose=False):
    t = time.time()
    yield
    elapsed = time.time() - t
    if label not in LABELS_TO_TIMES_AND_COUNTS:
        LABELS_TO_TIMES_AND_COUNTS[label] = [0., 0]
    LABELS_TO_TIMES_AND_COUNTS[label][0] += elapsed
    LABELS_TO_TIMES_AND_COUNTS[label][1] += 1
    if vervose:
        print(f"{label}: {elapsed * 1000:.1f} ms")


def print_time_and_reset(label):
    if label in LABELS_TO_TIMES_AND_COUNTS:
        total_time, count = LABELS_TO_TIMES_AND_COUNTS[label]
        print(f"{label} avg: {total_time / count * 1000:.1f} ms over {count} calls")
        LABELS_TO_TIMES_AND_COUNTS[label] = [0, 0]


def reset_timers():
    LABELS_TO_TIMES_AND_COUNTS.clear()


def print_all_timers():
    for label, (total_time, count) in LABELS_TO_TIMES_AND_COUNTS.items():
        print(f"{label:50} avg: {total_time / count * 1000:.1f} ms over {count} calls")
    LABELS_TO_TIMES_AND_COUNTS.clear()
