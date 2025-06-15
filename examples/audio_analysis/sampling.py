import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
Fs = 16000  # Sampling frequency
Ts = 1 / Fs  # Sampling period
time = np.arange(0, 0.1, Ts)  # Sampling lasts for 0.1 seconds

# Frequencies of the tones
freqs = [250, 550, 900]

# Generate tones and sum them
Xs = np.zeros((len(freqs), len(time)))
for i, freq in enumerate(freqs):
    Xs[i, :] = np.cos(2 * np.pi * freq * time)

# Final signal: sum of all tones
x = np.sum(Xs, axis=0)
x = x / np.max(np.abs(x))  # Normalize

# Plotting
plt.figure()
plt.plot(time, x)
plt.axis([0, time[-1], -1, 1])
plt.xlabel('Time (sec)')
plt.ylabel('Signal Amplitude')
plt.title('A simple audio signal')
plt.grid(True)
plt.show()
