# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 21:48:00 2026

@author: neevb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 20:37:43 2026

@author: neevb
"""

# TEST STRING = THIS IS A SAMPLE STRING FOR OFDM DEMO
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
sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
sdr.gain_control_mode_chan1 = "manual"  # manual or slow_attack
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


#%% Load Buffer and Transmit

sdr._ctx.set_timeout(0)
sdr.tx([tx_buffer*0.8, tx_buffer*0.8])  # Continuous transmission due to tx_cyclic_buffer = True

print("\n--- Starting Real-Time Reception ---")
print("Press Ctrl+C to stop.")

#%% Real-Time Setup and Plotting Configuration

# Calculate the transmitted frequency domain baseline (for plotting)
# We'll take the first data symbol to represent the Tx frequency structure
tx_sample_sym = tx_signal[len(pre_wave) : len(pre_wave) + N_FFT + 32]
tx_freq_mag = np.abs(np.fft.fft(tx_sample_sym[32:]))

plt.ion() # Enable interactive mode
fig, (ax_const, ax_freq) = plt.subplots(1, 2, figsize=(14, 6))

# 1. Constellation Plot Initialization
const_plot, = ax_const.plot([], [], 'r.', markersize=4, alpha=0.7)
ax_const.set_xlim(-2, 2)
ax_const.set_ylim(-2, 2)
ax_const.axhline(0, color='k', linewidth=0.5)
ax_const.axvline(0, color='k', linewidth=0.5)
ax_const.set_title("Real-Time 16-QAM Constellation")
ax_const.set_xlabel("In-Phase (I)")
ax_const.set_ylabel("Quadrature (Q)")
ax_const.grid(True)

# 2. Frequency Plot Initialization
# Pre-fill the X axis with 128 subcarrier indices and Y with zeros
freq_rx_plot, = ax_freq.plot(np.arange(N_FFT), np.zeros(N_FFT), 'b-', label='Received Magnitude')
freq_tx_plot, = ax_freq.plot(np.arange(N_FFT), tx_freq_mag, 'g--', label='Transmitted (Ref)', alpha=0.5)
ax_freq.set_xlim(0, N_FFT)
ax_freq.set_ylim(0, max(tx_freq_mag) * 1.5) # Initial Y limit
ax_freq.set_title("Frequency Domain Symbols")
ax_freq.set_xlabel("Subcarrier Index")
ax_freq.set_ylabel("Magnitude")
ax_freq.grid(True)
ax_freq.legend()

plt.tight_layout()

# Pre-calculate the low-pass filter to save CPU cycles inside the loop
nyquist = Sample_rate / 2
b, a = signal.butter(5, 4.5e6 / nyquist, btype='low')

symbol_len = N_FFT + 32  # 160 samples
num_payload_symbols = int(np.ceil(len(QAM_symbols) / (N_FFT - 12)))

#%% Real-Time Loop

# Define the Data Indices (Calculate ONCE outside the loop)
left_guard = np.arange(0, 6)
center_guard = np.array([N_FFT//2]) # Carrier 64
right_guard = np.arange(N_FFT - 5, N_FFT)
guard_indices = np.concatenate([left_guard, center_guard, right_guard])
all_indices = np.arange(N_FFT)
occupied_indices = np.concatenate([guard_indices, p_idx])
data_indices = np.setdiff1d(all_indices, occupied_indices)

try:
    while True:
        # 1. Receive data Buffer
        rx_raw = sdr.rx()
        rx_combined = rx_raw[0] + rx_raw[1]

        # 2. Low-Pass Filter
        rx_final = signal.filtfilt(b, a, rx_combined)

        # 3. Fast Synchronization (Cross-correlation only)
        # We drop the adaptive lag search to maintain real-time performance
        correlation = np.abs(np.correlate(rx_final, pre_wave, mode='same'))
        peak_idx = np.argmax(correlation)
        start_idx = peak_idx - len(pre_wave)//2

        # Safety check: ensure we have enough buffer left for the payload
        if start_idx < 0 or (start_idx + len(pre_wave) + num_payload_symbols * symbol_len > len(rx_final)):
            continue # Skip frame if preamble is too close to the edge

        # 4. Decoding & Equalization Setup
        payload_start = start_idx + len(pre_wave)
        received_symbols = []
        rx_freq_magnitudes = [] # To average for the frequency plot

        for i in range(num_payload_symbols):
            idx = payload_start + (i * symbol_len)
            symbol_with_cp = rx_final[idx : idx + symbol_len]
            
            if len(symbol_with_cp) == symbol_len:
                symbol_no_cp = symbol_with_cp[32:]
                sym_freq = np.fft.fft(symbol_no_cp)
                received_symbols.append(sym_freq)
                rx_freq_magnitudes.append(np.abs(sym_freq))

        if not received_symbols:
            continue

        equalized_data_symbols = []

        for sym_freq in received_symbols:
            # Channel Estimation and Interpolation
            received_pilots = sym_freq[p_idx]
            H_pilots = received_pilots / p_val
            interp_func = interp1d(p_idx, H_pilots, kind='linear', fill_value="extrapolate")
            H_estimated = interp_func(np.arange(N_FFT))
            
            # Equalization
            equalized_sym = sym_freq / (H_estimated + 1e-12)
            data_only = equalized_sym[data_indices]
            equalized_data_symbols.append(data_only)

        all_received_symbols = np.concatenate(equalized_data_symbols)

        # 5. Demapping to Bits
        recovered_bits = []
        for sym in all_received_symbols:
            closest_symbol = min(de_mapping_table_Qm.keys(), key=lambda x: abs(x - sym))
            recovered_bits.extend(de_mapping_table_Qm[closest_symbol])

        # Convert Bits to Text (Reusing your function logic)
        bytes_list = []
        for i in range(0, len(recovered_bits), 8):
            byte_chunk = recovered_bits[i:i+8]
            if len(byte_chunk) < 8: break
            byte_val = int("".join(map(str, byte_chunk)), 2)
            if byte_val == 0: break
            bytes_list.append(byte_val)
            
        final_message = bytes(bytes_list).decode('utf-8', errors='replace')
        
        # Overwrite the console line so it doesn't flood the terminal
        sys.stdout.write(f"\rReal-Time Decode: {final_message[:60]:<60}")
        sys.stdout.flush()

        # 6. Update Real-Time Plots
        # Update Constellation
        const_plot.set_xdata(all_received_symbols.real)
        const_plot.set_ydata(all_received_symbols.imag)
        
        # Update Frequency Magnitude (Average across all payload symbols in this frame)
        avg_rx_freq = np.mean(rx_freq_magnitudes, axis=0)
        freq_rx_plot.set_ydata(avg_rx_freq)
        
        # Dynamically scale Y-axis for frequency if signal gets hot
        current_ymax = ax_freq.get_ylim()[1]
        max_val = np.max(avg_rx_freq)
        if max_val > current_ymax or max_val < current_ymax * 0.5:
            ax_freq.set_ylim(0, max_val * 1.5)

        # Draw the frame
        fig.canvas.draw()
        fig.canvas.flush_events()

except KeyboardInterrupt:
    print("\nExecution stopped by user.")

finally:
    #%% Destroy Tx Buffer and cleanup
    sdr.tx_destroy_buffer()
    plt.ioff()
    plt.show()
    print("Buffer destroyed and plotting closed.")