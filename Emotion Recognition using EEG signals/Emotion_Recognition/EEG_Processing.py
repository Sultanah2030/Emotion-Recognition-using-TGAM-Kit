import time
import numpy as np
import csv
from datetime import datetime
import os
from scipy import signal
from scipy.stats import entropy
from scipy.signal.windows import hamming
from antropy import perm_entropy  
from NeuroPy import NeuroPy # This function uses code from the NeuroPy library (Copyright (c) 2013, Sahil Singh)

class EEGProcessor:
    def __init__(self, port="COM3", baudRate=57600):
        # Initialize EEG device TGAM connection and parameters
        self.device = NeuroPy(port, baudRate)
        self.buffer = []
        self.BUFFER_SIZE = 512 # Number of EEG samples to hold before processing
        self.SAMPLING_RATE = 256 # EEG signal sampling rate in Hz
        self.filename = 'SadState7.csv'
        self.freq_bands = {
            'Delta': (1,4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        } #initialize frequency Bands for each band

        self.init_csv() # Setup output CSV with headers

    def init_csv(self):
        # Create directory and CSV file for saving features if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
        header = ['timestamp', 'spectral_entropy', 'permutation_entropy']
        for band in self.freq_bands:
            header.extend([
                f'{band}_psd', 
                f'{band}_meanpsd', 
                f'{band}_mean', 
                f'{band}_std', 
                f'{band}_power',
                f'{band}_freqs'
            ])
        header.append('power_ratio')
        header.append('label')
        if not os.path.exists(f'data/{self.filename}'):
            with open(f'data/{self.filename}', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def save_raw_and_filtered_data(self, raw_data, filtered_data, label):
        # Save both raw and filtered EEG signal samples to a file for visualization
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        rows = []
        for i, val in enumerate(raw_data):
            rows.append([timestamp, 'raw', i, val])
        for i, val in enumerate(filtered_data):
            rows.append([timestamp, 'filtered', i, val])

        filepath = 'data/rawvsfiltered_wdelta.csv'
        file_exists = os.path.exists(filepath)

        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
               writer.writerow(['timestamp', 'type', 'index', 'value'])  # header
            writer.writerows(rows)

    def bandpass_filter(self, data, band):
        # Filter EEG data to isolate one frequency band (theta, alpha, beta, etc.)
        nyquist = 0.5 * self.SAMPLING_RATE # Calculate Nyquist frequency, which is half the sampling rate
        low, high = band # Get the low and high cutoff frequencies for the bandpass filter
         # Create a Butterworth filter with order 8 for the given frequency range
        b, a = signal.butter(8, [low / nyquist, high / nyquist], btype='band')
        # Apply the filter to the EEG data using zero-phase filtering to prevent phase distortion
        return signal.filtfilt(b, a, data)
    

    def process_band(self, data, band):
        # Extract frequency-domain features from a specific band using Welch method
        filtered = self.bandpass_filter(data, band)
        window = hamming(len(filtered))
        windowed_data = filtered * window
        freqs, psd = signal.welch(windowed_data, fs=self.SAMPLING_RATE, nperseg=256, noverlap=128)
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        band_psd = psd[band_mask]
        band_freqs = freqs[band_mask]
        mean_psd = np.mean(band_psd)
        mean_val = np.mean(filtered)
        std_val = np.std(filtered)
        band_power = np.trapz(band_psd, band_freqs)
        return band_psd, mean_psd, mean_val, std_val, band_power,band_freqs

    def apply_moving_median(self, data, kernel_size=5):
        filtered = []
        half_k = kernel_size // 2
        for i in range(len(data)):
            window = data[max(0, i - half_k):min(len(data), i + half_k + 1)]
            median = np.median(window)
            filtered.append(median)
        return np.array(filtered)
    
    def notch_filter(self, data, freq=60.0, Q=30.0):
        nyquist = 0.5 * self.SAMPLING_RATE
        w0 = freq / nyquist
        b, a = signal.iirnotch(w0, Q)
        return signal.filtfilt(b, a, data)


    def process_buffer(self):
        # Main pipeline that applies preprocessing, extracts features, and saves them
        data = np.array(self.buffer[-self.BUFFER_SIZE:])
        raw_data = data.copy()
        # Step 1: Apply moving median to remove sudden spikes and small artifacts
        data = self.apply_moving_median(data)
        # Step 2: Apply notch filter to remove 60Hz power line interference
        data = self.notch_filter(data)
        # Step 3: Thresholding - reject data with extreme values
        if np.any(np.abs(data) > 150):
            return
        
        self.save_raw_and_filtered_data(raw_data, data, label=0)

        # Window the signal for frequency analysis
        window = hamming(len(data))
        windowed_data = data * window

        # Extract entropy features (complexity of signal)
        freqs, psd = signal.welch(windowed_data, fs=self.SAMPLING_RATE, nperseg=256, noverlap=128)
        spectral_entropy = entropy(psd / np.sum(psd))
        perm_ent = perm_entropy(windowed_data, normalize=True)

        features = [spectral_entropy, perm_ent]
        power_values = {}

        # Step 4: Extract features for each frequency band
        for band_name, band_range in self.freq_bands.items():
            band_psd, mean_psd, mean_val, std_val, power, band_freqs = self.process_band(data, band_range)
            features.extend([band_psd.tolist(), mean_psd, mean_val, std_val, power,band_freqs.tolist()])
            power_values[band_name] = power

        # Step 5: Compute Alpha/Beta power ratio (used to classify emotional state)
        alpha = power_values.get('alpha', 1)
        beta = power_values.get('beta', 0)
        power_ratio = alpha / beta if beta != 0 else 0

        features.append(power_ratio)
        features.append(0)  # label for supervised learning (0 = sad , 1 = Happy)

        self.save_features(features)

    def save_features(self, features):
        # Save the extracted features into the main dataset file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        row = [timestamp] + features
        with open(f'data/{self.filename}', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def start(self):
        # Begin EEG data collection from NeuroSky device
        print(f"Attempting to connect to {self.device._NeuroPy__serialPort} at {self.device._NeuroPy__serialBaudRate} baud")
        self.device.setCallBack("rawValue", self.raw_callback) # This function uses code from the NeuroPy library (Copyright (c) 2013, Sahil Singh)
        self.device.start()
        print("Connection started. Waiting for data...")

    def stop(self):
        # Stop EEG data collection
        self.device.stop()

    def raw_callback(self, value):
        # Collect raw EEG values in a buffer and trigger processing when full
        self.buffer.append(value)
        if len(self.buffer) >= self.BUFFER_SIZE:
            self.process_buffer()
            self.buffer = self.buffer[self.BUFFER_SIZE // 2:]

if __name__ == "__main__":
    # Run the program and start collecting EEG data until manually stopped
    processor = EEGProcessor()
    processor.start()
    print("Collecting EEG data. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        processor.stop()
        print("Data collection stopped.")
