import time
import numpy as np
import csv
from datetime import datetime
import os
from scipy import signal
from scipy.stats import entropy
from scipy.signal.windows import hamming
from antropy import perm_entropy  
from NeuroPy import NeuroPy

class EEGProcessor:
    def __init__(self, port="COM3", baudRate=57600):
        self.device = NeuroPy(port, baudRate)
        self.buffer = []
        self.BUFFER_SIZE = 512
        self.SAMPLING_RATE = 256
        self.filename = 'SadState7.csv'
        self.freq_bands = {
            'Delta': (1,4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        self.init_csv()

    def init_csv(self):
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
        nyquist = 0.5 * self.SAMPLING_RATE
        low, high = band
        b, a = signal.butter(8, [low / nyquist, high / nyquist], btype='band')
        return signal.filtfilt(b, a, data)
    

    def process_band(self, data, band):
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
        data = np.array(self.buffer[-self.BUFFER_SIZE:])
        raw_data = data.copy()
        data = self.apply_moving_median(data)
        data = self.notch_filter(data)

        if np.any(np.abs(data) > 150):
            return
        
        self.save_raw_and_filtered_data(raw_data, data, label=0)

        window = hamming(len(data))
        windowed_data = data * window

        freqs, psd = signal.welch(windowed_data, fs=self.SAMPLING_RATE, nperseg=256, noverlap=128)
        spectral_entropy = entropy(psd / np.sum(psd))
        perm_ent = perm_entropy(windowed_data, normalize=True)

        features = [spectral_entropy, perm_ent]
        power_values = {}

        for band_name, band_range in self.freq_bands.items():
            band_psd, mean_psd, mean_val, std_val, power, band_freqs = self.process_band(data, band_range)
            features.extend([band_psd.tolist(), mean_psd, mean_val, std_val, power,band_freqs.tolist()])
            power_values[band_name] = power

        alpha = power_values.get('alpha', 1)
        beta = power_values.get('beta', 0)
        power_ratio = alpha / beta if beta != 0 else 0

        features.append(power_ratio)
        features.append(0)  # label

        self.save_features(features)

    def save_features(self, features):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        row = [timestamp] + features
        with open(f'data/{self.filename}', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def start(self):
        print(f"Attempting to connect to {self.device._NeuroPy__serialPort} at {self.device._NeuroPy__serialBaudRate} baud")
        self.device.setCallBack("rawValue", self.raw_callback)
        self.device.start()
        print("Connection started. Waiting for data...")

    def stop(self):
        self.device.stop()

    def raw_callback(self, value):
        self.buffer.append(value)
        #print(f"Raw value received: {value} (Buffer size: {len(self.buffer)})")  # Debug
        if len(self.buffer) >= self.BUFFER_SIZE:
            #print("Processing buffer")  # Debug
            self.process_buffer()
            self.buffer = self.buffer[self.BUFFER_SIZE // 2:]

if __name__ == "__main__":
    processor = EEGProcessor()
    processor.start()
    print("Collecting EEG data. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        processor.stop()
        print("Data collection stopped.")
