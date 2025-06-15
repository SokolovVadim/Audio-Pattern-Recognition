import numpy as np
import scipy.io.wavfile as wavfile
import simpleaudio as sa
import tempfile
import os

def play_sound(x, fs=8000, nbits=16):
    # Normalize to 16-bit PCM range
    x_int = np.int16(x * 32767)

    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wavfile.write(f.name, fs, x_int)

        try:
            # Playback using simpleaudio
            wave_obj = sa.WaveObject.from_wave_file(f.name)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        finally:
            os.remove(f.name)

# Example usage:
if __name__ == "__main__":
    Fs = 16000
    duration = 0.1
    t = np.linspace(0, duration, int(Fs * duration), endpoint=False)
    x = np.cos(2 * np.pi * 440 * t)  # A simple 440 Hz tone
    x /= np.max(np.abs(x))  # Normalize
    play_sound(x, Fs)
