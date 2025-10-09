import numpy as np
import sounddevice as sd


def beep(frequency=880, duration=0.05, samplerate=44100, amplitude=0.3):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    sd.play(wave, samplerate)
    sd.wait()  # Wait until the sound has finished playing
