# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 20:37:43 2026

@author: neevb
"""
#%% Import Libraries

import numpy as np
import scipy as scp
import pickle
import matplotlib.pyplot as plt
import adi
import os
import time
import cupy as cp
import cupyx.scipy.signal
import signal
import sys
import Adaptive_threshold
import helpers
from scipy import signal
from scipy.interpolate import interp1d


print(" All Libraries Imported!!!")

#%% Constants

Sample_rate = int(2500000*4)   # (2.5Msps)
SDR_freq = int(2.2e9)      # (2.2GHz)
interfere_signal_freq = pickle.load(open("hb100_freq_val.pkl", "rb"))
d = 0.014                  # element to element spacing of the antenna
nSamples = int(2**18)        # Sample buffer size
SDR_rx_gain = int(30)      # 20dB, must be between -3 and 70
ADF4159_freq = int(SDR_freq + interfere_signal_freq)

#%% Setup Devices

sdr_ip = "ip:192.168.2.1"
rpi_ip = "ip:phaser.local"

sdr = adi.ad9361(uri=sdr_ip)
phaser = adi.CN0566(uri=rpi_ip, sdr=sdr)

print("Phaser And PlutoSDR connected")
   
time.sleep(0.05)

# Initialize both ADAR1000s, and load calibration weights
phaser.configure(device_mode="rx")
phaser.load_gain_cal()
phaser.load_phase_cal()

# Setup Raspberry Pi GPIO states
phaser._gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
phaser._gpios.gpio_vctrl_1 = 1  # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
phaser._gpios.gpio_vctrl_2 = (
        1  # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)
    )

# Configure SDR Rx
sdr.sample_rate = int(Sample_rate)
sdr.rx_lo = int(SDR_freq)               # set this to ADF4159_freq - (the freq of the HB100)
sdr.rx_enabled_channels = [0, 1]        # enable Rx1 (voltage0) and Rx2 (voltage1)
sdr.rx_buffer_size = int(nSamples)
sdr.gain_control_mode_chan0 = "slow_attack"  # manual or slow_attack
sdr.gain_control_mode_chan1 = "slow_attack"  # manual or slow_attack
sdr.rx_hardwaregain_chan0 = int(SDR_rx_gain)  # must be between -3 and 70
sdr.rx_hardwaregain_chan1 = int(SDR_rx_gain)  # must be between -3 and 70
sdr.rx_rf_bandwidth = int(Sample_rate * 2)    # Wider filter

# Configure SDR Tx
sdr.tx_lo = int(2.2e9)
sdr.tx_enabled_channels = [0, 1]
sdr.tx_cyclic_buffer = True             # must set cyclic buffer to true for the tdd burst mode
sdr.tx_hardwaregain_chan0 = -10         # must be between 0 and -88
sdr.tx_hardwaregain_chan1 = -10          # must be between 0 and -88; Chanel 1 is set to transmit
sdr.tx_rf_bandwidth = int(Sample_rate * 2) # Wider filter

# Configure Phaser mode
BW = 500e6
num_steps = 1000
ramp_time = 1e3  # us
ramp_time_s = ramp_time / 1e6
phaser.frequency = int(ADF4159_freq / 4)  # Output frequency divided by 4
phaser.freq_dev_range = int(
    BW / 4
)  # frequency deviation range in Hz.  This is the total freq deviation of the complete freq ramp
phaser.freq_dev_step = int(
    (BW / 4) / num_steps
)  # frequency deviation step in Hz.  This is fDEV, in Hz.  Can be positive or negative
phaser.freq_dev_time = int(
    ramp_time
)  # total time (in us) of the complete frequency ramp
print("requested freq dev time = ", ramp_time)
print("actual freq dev time = ", phaser.freq_dev_time)
phaser.delay_word = 4059  # 12 bit delay word.  4095*PFD = 40.95 us.  For sawtooth ramps, this is also the length of the Ramp_complete signal
phaser.delay_clk = "PFD"  # can be 'PFD' or 'PFD*CLK1'
phaser.delay_start_en = 0  # delay start
phaser.ramp_delay_en = 0  # delay between ramps.
phaser.trig_delay_en = 0  # triangle delay
phaser.ramp_mode = "disabled"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
phaser.sing_ful_tri = (
    0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
)
phaser.tx_trig_en = 0  # start a ramp with TXdata
phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

#%% Applying GA weights

weights = [-0.1650 + 0.3488j,
           -0.1986 + 0.4200j,
           -0.2348 + 0.3917j,
           -0.1577 + 0.5199j,
           -0.2975 + 0.4452j,
           -0.3067 + 0.3384j,
           -0.2301 + 0.4305j,
           -0.2580 + 0.3478j]
weights = np.array(weights)
weights_mag = np.round(np.abs(weights)*127)
weights_degrees = np.round(np.rad2deg(np.angle(weights))/2.8125)*2.8125
for i in range(0, 8):
    phaser.set_chan_phase(i, 0, apply_cal=True)

for i in range(0, 8):
    phaser.set_chan_gain(i, 127, apply_cal=True)
    
phaser.latch_rx_settings()

#%% Generate QAM Map

Qm = 4              # bits/symbol
mapping_table_Qm, de_mapping_table_Qm = helpers.mapping_table(Qm, plot=True) # mapping table for Qm

#%% Generate Baseband Data and Corresponding Symbols

data_string = str(input("Enter the Data String: "))
bit_sequence = helpers.string_to_bits(data_string)
QAM_symbols = helpers.map_bits_to_symbols(bit_sequence, mapping_table_Qm, Qm)

#%% Generate Pilot Symbols

N_FFT = 128
p_idx, p_val = helpers.generate_pilots(N_FFT, pilot_value=1.5+0j)

#%% Generate Preamble Symbols

pre_idx, pre_syms, pre_wave = helpers.generate_preamble_data(N_FFT)

#%% Final Assembly + Generate Tx buffer

# Create the OFDM signal
tx_signal = helpers.assemble_ofdm_buffer(QAM_symbols, p_idx, p_val, pre_wave)

# --- ADD SILENCE PADDING ---
# Add samples of zeros to create a gap between bursts
silence = np.zeros(500, dtype=complex)
tx_buffer = np.concatenate((tx_signal, silence))


#%% Load Buffer and Trasnmit

sdr._ctx.set_timeout(0)
sdr.tx([tx_buffer*0.8, tx_buffer*0.8])  # only send data to the 2nd channel (that's all we need)

plt.plot(tx_buffer)

#%% Receive data Buffer

rx_raw = sdr.rx()
rx_combined = rx_raw[0] + rx_raw[1]

# 3. Low-Pass Filter
nyquist = Sample_rate / 2
b, a = signal.butter(5, 4.5e6 / nyquist, btype='low')
rx_final = signal.filtfilt(b, a, rx_combined)

#%% Synchronization & Slicing 

# Perform cross-correlation with your known preamble
correlation = np.abs(np.correlate(rx_final, pre_wave, mode='same'))

# FIX: mode='same' places the peak in the CENTER of the preamble.
# We must shift back by half the preamble length to find the START.
peak_idx = np.argmax(correlation)
start_idx = peak_idx - len(pre_wave)//2

# Safety check to prevent negative indices
if start_idx < 0: start_idx = 0


# Extract the Payload
# We skip the preamble (160 samples) and start taking payload symbols
symbol_len = N_FFT + 32 # 160 samples
num_payload_symbols = int(np.ceil(len(QAM_symbols) / (N_FFT - 12))) # Calculate based on data

# Define the Data Indices (Must match helpers.py assemble_ofdm_buffer)
left_guard = np.arange(0, 6)
center_guard = np.array([N_FFT//2]) # Carrier 64
right_guard = np.arange(N_FFT - 5, N_FFT)
guard_indices = np.concatenate([left_guard, center_guard, right_guard])
all_indices = np.arange(N_FFT)
occupied_indices = np.concatenate([guard_indices, p_idx])
data_indices = np.setdiff1d(all_indices, occupied_indices)

# ADAPTIVE TIMING SEARCH
print("\n--- Starting Adaptive Timing Scan ---")
best_score = float('inf')
best_lag = 0
search_range = range(-20, 21)  # Scan +/- 20 samples

# We use a temporary loop to test synchronization "crispness"
for lag in search_range:
    trial_start = start_idx + lag
    temp_eq_symbols = []
    
    # Test only the first 20 symbols to save time
    test_iterations = min(20, num_payload_symbols)
    current_payload_start = trial_start + symbol_len
    
    valid_lag = True
    for i in range(test_iterations):
        idx = current_payload_start + (i * symbol_len)
        if idx + symbol_len > len(rx_final): 
            valid_lag = False
            break
            
        # Slice and FFT
        sym_time = rx_final[idx : idx+symbol_len]
        sym_freq = np.fft.fft(sym_time[32:]) # Remove CP
        
        # LS Equalization (Quick & Dirty for metric)
        H_LS = sym_freq[p_idx] / p_val
        interp = interp1d(p_idx, H_LS, kind='linear', fill_value="extrapolate")
        H_est = interp(np.arange(N_FFT))
        eq_sym = sym_freq / (H_est + 1e-12)
        
        # Store Data Subcarriers
        temp_eq_symbols.extend(eq_sym[data_indices])

    if not valid_lag or len(temp_eq_symbols) == 0: continue

    # CALCULATE SCORE (EVM / Cluster Tightness)
    # Normalize points roughly to unity
    points = np.array(temp_eq_symbols)
    avg_mag = np.mean(np.abs(points))
    norm_points = points / (avg_mag + 1e-12)
    
    # Measure variance from nearest integers (grid alignment)
    # 16-QAM usually maps to +/-1 and +/-3. We check modulo 2 variance.
    score = np.var(np.abs(norm_points.real) % 2) + np.var(np.abs(norm_points.imag) % 2)
    
    print(f"Lag {lag}: Score {score:.4f}")
    
    if score < best_score:
        best_score = score
        best_lag = lag

print(f"--> Best Time Offset Found: {best_lag}")
final_start_idx = start_idx + best_lag

#%% Decoding
symbol_len = N_FFT + 32  # 160 samples
payload_start = final_start_idx + symbol_len  # Skip preamble
received_symbols = []

samples_remaining = len(rx_final) - payload_start
max_symbols_possible = samples_remaining // symbol_len

# Use the smaller of the two to prevent slicing empty data
actual_iterations = min(num_payload_symbols, max_symbols_possible)

for i in range(actual_iterations):
    idx = payload_start + (i * symbol_len)
    symbol_with_cp = rx_final[idx : idx + symbol_len]
    
    # Final check to ensure we have a full 160 samples
    if len(symbol_with_cp) == symbol_len:
        symbol_no_cp = symbol_with_cp[32:]
        symbol_freq = np.fft.fft(symbol_no_cp)
        received_symbols.append(symbol_freq)
    else:
        print(f"Warning: Symbol {i} was truncated. Stopping.")
        break


equalized_data_symbols = []

for sym_freq in received_symbols:
    # hannel Estimation at Pilot Locations
    # Formula: H = Received / Transmitted
    received_pilots = sym_freq[p_idx]
    H_pilots = received_pilots / p_val
    
    # Channel Interpolation
    # We estimate the channel for all 128 subcarriers by interpolating between the 8 pilots
    # 'linear' interpolation handles the phase slope across the frequency band
    interp_func = interp1d(p_idx, H_pilots, kind='linear', fill_value="extrapolate")
    H_estimated = interp_func(np.arange(N_FFT))
    
    # Equalization (Zero-Forcing)
    # Divide the received frequency symbol by the estimated channel
    # We add a tiny 1e-12 to avoid division by zero in the guard bands
    equalized_sym = sym_freq / (H_estimated + 1e-12)
    
    # Extract only the Data subcarriers (removing pilots and guards)
    data_only = equalized_sym[data_indices]
    equalized_data_symbols.append(data_only)

final_points = []
final_points = np.concatenate(equalized_data_symbols)
plt.figure(figsize=(8,8))
plt.plot(final_points.real, final_points.imag, 'r.', markersize=2)
plt.axhline(0, color='k', linewidth=1)
plt.axvline(0, color='k', linewidth=1)
plt.title("Final Equalized 16-QAM Constellation")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.grid(True)
plt.show()

# Flatten all equalized symbols into one array to find the global scale
all_eq_points = np.concatenate(equalized_data_symbols)

print("\n--- Starting Decoding ---")

# Flatten the list of symbol arrays into one long stream of symbols
all_received_symbols = np.concatenate(equalized_data_symbols)

# De-map Symbols to Bits
recovered_bits = []

for symbol in all_received_symbols:
    # Find the closest point in the constellation (Minimum Distance)
    # This checks every point in your QAM map to find the best match
    closest_symbol = min(de_mapping_table_Qm.keys(), key=lambda x: abs(x - symbol))
    
    # Retrieve the bits associated with that symbol
    bits = de_mapping_table_Qm[closest_symbol]
    recovered_bits.extend(bits)

# Convert Bits to Text
def bits_to_text(bit_array):
    # Group bits into bytes (8 bits per character)
    bytes_list = []
    for i in range(0, len(bit_array), 8):
        byte_chunk = bit_array[i:i+8]
        
        # Stop if we have an incomplete byte at the end
        if len(byte_chunk) < 8:
            break
            
        # Convert [0, 1, 0, ...] to integer
        byte_val = int("".join(map(str, byte_chunk)), 2)
        
        # Optional: Stop at a null terminator (if you sent one)
        if byte_val == 0: 
            break
            
        bytes_list.append(byte_val)
        
    # Decode bytes to string, replacing errors with '?'
    return bytes(bytes_list).decode('utf-8', errors='replace')

# Print the Result
final_message = bits_to_text(recovered_bits)
print(f"Decoded Message: {final_message}")
print("-------------------------")

#%% Destroy Tx Buffer

sdr.tx_destroy_buffer()