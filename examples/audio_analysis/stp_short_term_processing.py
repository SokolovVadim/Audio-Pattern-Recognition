import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal.windows import hamming

def stp_file(wav_file, window_length_sec, step_sec):
    # Read WAV file
    fs, x = wav.read(wav_file)
    
    # Normalize if necessary (e.g. for int16 input)
    if x.dtype != np.float32 and x.dtype != np.float64:
        x = x / np.max(np.abs(x))  # normalize to [-1, 1]

    # If stereo, convert to mono
    if x.ndim == 2:
        x = np.mean(x, axis=1)

    # Convert window and step from seconds to samples
    window_length = int(round(window_length_sec * fs))
    step = int(round(step_sec * fs))

    cur_pos = 0
    L = len(x)
    num_of_frames = int(np.floor((L - window_length) / step)) + 1

    for i in range(num_of_frames):
        frame = x[cur_pos:cur_pos + window_length]
        frameW = frame * hamming(len(frame), sym=False)

        # Plot original and windowed frame
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(frame)
        plt.title("Current frame (original)")
        plt.ylim([-1, 1])

        plt.subplot(2, 1, 2)
        plt.plot(frameW)
        plt.title("Current frame (windowed)")
        plt.ylim([-1, 1])

        plt.pause(0.1)
        cur_pos += step

    plt.show()

# Example usage
if __name__ == "__main__":
    stp_file("scottish.wav", window_length_sec=0.03, step_sec=0.01)  # 30ms window, 10ms step
