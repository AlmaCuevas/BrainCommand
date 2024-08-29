import numpy as np
from matplotlib import pyplot as plt
from processing_eeg_methods.data_utils import data_normalization
from scipy import signal

def check_eeg(eeg, t_eeg, t_marker_start, t_marker_end):
    if eeg:

        x_array = np.array([eeg])
        x_array = x_array[:, :,
                  :-9]  # The last channels are acceleromet
        # er (x3), gyroscope (x3), validity, battery and counter
        x_array = np.transpose(x_array, (0, 2, 1))
        x_array = signal.detrend(x_array)
        x_array = data_normalization(x_array)

        x_array = np.transpose(x_array, (0, 1, 2))
        plt.clf()
        plt.plot(t_eeg, x_array[0, 1, :], label='EEG Signal')

        # Calculate and plot the average line
        avg_line = np.mean(x_array[0, 7, :])
        plt.axhline(y=avg_line, color='r', linestyle='--', label='Average Line')

        plt.axvline(x=t_marker_start, color='g', linestyle='--', label='Start Marker')
        plt.axvline(x=t_marker_end, color='b', linestyle='--', label='End Marker')

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

