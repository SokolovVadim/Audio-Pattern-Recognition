import numpy as np
import matplotlib.pyplot as plt
import simpleaudio as sa

# Sampling parameters
Fs = 16000
Ts = 1 / Fs
time = np.arange(0, 0.1, Ts)  # 0.1 second duration

# Left and right channel signals
xLeft = np.cos(2 * np.pi * 250 * time)  # 250 Hz
xRight = np.cos(2 * np.pi * 450 * time)  # 450 Hz

# Stack into a stereo signal (shape: [samples, 2])
x = np.column_stack((xLeft, xRight))

# Plot both channels
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(time, x[:, 0])
plt.title("Left Channel")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(time, x[:, 1])
plt.title("Right Channel")
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Normalize and convert to 16-bit PCM
x_norm = x / np.max(np.abs(x))
x_int16 = np.int16(x_norm * 32767)

# Playback using simpleaudio
play_obj = sa.play_buffer(x_int16, num_channels=2, bytes_per_sample=2, sample_rate=Fs)
play_obj.wait_done()
