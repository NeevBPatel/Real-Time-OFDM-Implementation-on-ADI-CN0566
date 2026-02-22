# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 21:32:41 2026

@author: neevb
"""

import numpy as np
import matplotlib.pyplot as plt

# Configuration variables
save_plots = False
plot_width = 8
titles = True

def mapping_table(Qm, plot=False):
    """
    Create a modulation mapping table and its inverse for an OFDM system.
    """
    # Size of the constellation
    size = int(np.sqrt(2**Qm))
    
    # Create the constellation points
    a = np.arange(size, dtype=np.float32)
    
    # Shift the constellation to the center
    b = a - np.mean(a)
    
    # Use broadcasting to create the complex constellation grid
    C = (b[:, None] + 1j * b).flatten()
    
    # Normalize the constellation (RMS value)
    C /= np.sqrt(np.mean(np.abs(C)**2))
    
    # Function to convert index to binary
    def index_to_binary(i, Qm):
        return tuple(map(int, '{:0{}b}'.format(int(i), Qm)))
    
    # Create the mapping dictionary
    mapping = {index_to_binary(i, Qm): val for i, val in enumerate(C)}
    
    # Create the demapping table
    demapping = {v: k for k, v in mapping.items()}

    # Plot the constellation if plot is True
    if plot:
        plt.figure(figsize=(4, 4))
        plt.scatter(C.real, C.imag)
        
        if titles:
            plt.title(f'Constellation - {Qm} bits per symbol')
        
        plt.grid(True)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.xlabel("In-Phase (I)")
        plt.ylabel("Quadrature (Q)")
        plt.show()
        
        if save_plots:
            plt.savefig('pics/const.png', bbox_inches='tight')
    
    return mapping, demapping


def string_to_bits(input_string):
    """
    Converts a string to a binary sequence and zero-pads it to 432 bits.
    
    Args:
        input_string (str): The text to convert.
        
    Returns:
        numpy.array: An array of 0s and 1s with length 432.
    """
    # Convert string to bytes
    byte_data = input_string.encode('utf-8')
    
    # Convert bytes to list of bits
    # standard "f-string" formatting: '08b' means 8 bits, zero-padded
    bit_string = ''.join([f'{byte:08b}' for byte in byte_data])
    
    # Check length
    current_length = len(bit_string)
    target_length = 432
    
    if current_length > target_length:
        print(f"Warning: Input string is too long! Truncating {current_length} bits to {target_length}.")
        bit_string = bit_string[:target_length]
    else:
        # Zero Pad (Append '0's to the end)
        padding_needed = target_length - current_length
        bit_string += '0' * padding_needed
        
    # Convert to numpy array of integers
    bit_array = np.array([int(b) for b in bit_string], dtype=int)
    
    return bit_array

def map_bits_to_symbols(bit_sequence, mapping_dict, Qm):
    """
    Maps a binary sequence to complex symbols using the provided mapping dictionary.
    
    Args:
        bit_sequence (np.array): 1D array of bits (0s and 1s).
        mapping_dict (dict): The dictionary returned by mapping_table().
        Qm (int): Bits per symbol (4 for 16-QAM).
        
    Returns:
        np.array: Array of complex symbols.
    """
    # Safety Check: Ensure bit count is divisible by Qm
    if len(bit_sequence) % Qm != 0:
        raise ValueError(f"Error: Bit sequence length ({len(bit_sequence)}) is not divisible by {Qm}.")
        
    # Reshape bits into chunks of 'Qm' (Rows of 4 bits)
    bit_groups = bit_sequence.reshape(-1, Qm)
    
    # Convert to Symbols
    symbols = np.array([mapping_dict[tuple(group)] for group in bit_groups])
    
    return symbols

def generate_pilots(N_FFT, pilot_value=1.5+0j):
    """
    Generates pilot symbols and their indices for a 128-point OFDM system.
    
    Args:
        N_FFT (int): FFT size (e.g., 128).
        pilot_value (complex): The complex value of the pilot (boosted power).
        
    Returns:
        pilot_indices (np.array): The array indices (0 to N_FFT-1) where pilots go.
        pilot_symbols (np.array): The complex values of the pilots.
    """
    
    # Define Pilot Locations (Logical Indices)
    logical_locs = np.array([-42, -28, -14, -7, 7, 14, 28, 42])
    
    # Map to FFT Array Indices (0 to 127)
    pilot_indices = []
    for loc in logical_locs:
        if loc < 0:
            pilot_indices.append(N_FFT + loc)
        else:
            pilot_indices.append(loc)
            
    pilot_indices = np.array(sorted(pilot_indices))
    
    # Generate Symbols
    pilot_symbols = np.full(len(pilot_indices), pilot_value, dtype=complex)
    
    return pilot_indices, pilot_symbols

def generate_preamble_data(N_FFT):
    """
    Generates the frequency-domain symbols and indices for a Schmidl & Cox Preamble.
    
    Args:
        N_FFT (int): FFT size (e.g., 128).
        
    Returns:
        preamble_indices (np.array): The even indices (0, 2, 4...) used.
        preamble_symbols (np.array): The complex values at those indices.
        time_domain_preamble (np.array): The final wave to attach to your buffer (with CP).
    """
    
    # Fixed Seed (Crucial!)
    # The Receiver MUST know exactly what these random numbers are to detect them. Only used for Tx and Rx seperate codes,
    # not for this loopback system
    # np.random.seed(42) 
    
    # Generate Symbols for EVEN carriers only
    # We need N_FFT / 2 symbols because we skip every other bin.
    # We use high-power BPSK (1+1j or -1-1j) for robustness.
    # Logic: 0 -> -1-1j, 1 -> 1+1j
    bits = np.random.randint(0, 2, N_FFT // 2)
    bpsk_vals = (2*bits - 1) + 1j*(2*bits - 1)
    
    # Boost power slightly (e.g., amplitude 2.0) to make it stand out
    preamble_symbols = bpsk_vals * np.sqrt(2) 
    
    # Define Indices (0, 2, 4, 6... 126)
    preamble_indices = np.arange(0, N_FFT, 2)
    
    # Generate the Time Wave (for the buffer)
    # Construct the full frequency array
    freq_grid = np.zeros(N_FFT, dtype=complex)
    freq_grid[preamble_indices] = preamble_symbols
    
    # IFFT to Time Domain
    # This automatically creates the [A, A] repeating structure
    time_signal = np.fft.ifft(freq_grid)
    
    # Add Cyclic Prefix (Standard 32 samples)
    cp_length = 32
    time_domain_preamble = np.concatenate([time_signal[-cp_length:], time_signal])
    
    return preamble_indices, preamble_symbols, time_domain_preamble

def assemble_ofdm_buffer(QAM_symbols, p_idx, p_val, pre_wave, N_FFT=128, N_CP=32):
    """
    Assembles QAM symbols, pilots, and a preamble into a final time-domain OFDM frame.
    
    Args:
        QAM_symbols (np.array): 1D array of complex data symbols (e.g., 16-QAM).
        p_idx (np.array): Indices of pilot subcarriers.
        p_val (np.array): Values of pilot subcarriers.
        pre_wave (np.array): Time-domain preamble (already includes CP).
        N_FFT (int): FFT size.
        N_CP (int): Cyclic Prefix length.
        
    Returns:
        np.array: The final time-domain buffer scaled for PlutoSDR.
    """
    
    # Define Resource Map 
    # Guards: DC (0), Nyquist (N/2), and edges to prevent aliasing
    left_guard = np.arange(0, 6)   # 0, 1, 2, 3, 4, 5
    center_guard = np.array([N_FFT//2]) # Nyquist (64)
    right_guard = np.arange(N_FFT - 5, N_FFT) # 123, 124, 125, 126, 127
    
    guard_indices = np.concatenate([left_guard, center_guard, right_guard])
    
    # Determine Data Indices: All Indices - (Pilots + Guards)
    all_indices = np.arange(N_FFT)
    occupied_indices = np.concatenate([guard_indices, p_idx])
    
    # np.setdiff1d returns the sorted unique values in ar1 that are not in ar2
    data_indices = np.setdiff1d(all_indices, occupied_indices)  # returns remaining indices that can be used as data indices
    
    carriers_per_symbol = len(data_indices)
    
    # Segmentation (Splitting Data into Symbols)
    # Calculate how many OFDM symbols we need to carry all the QAM data
    num_ofdm_symbols = int(np.ceil(len(QAM_symbols) / carriers_per_symbol))
    
    payload_waves = []
    
    for i in range(num_ofdm_symbols):
        # Create empty frequency grid
        symbol_freq = np.zeros(N_FFT, dtype=complex)
        
        # Insert Pilots
        symbol_freq[p_idx] = p_val
        
        # Insert Data
        start_loc = i * carriers_per_symbol
        end_loc = start_loc + carriers_per_symbol
        
        # Extract chunk
        chunk = QAM_symbols[start_loc:end_loc]
        
        # Handle Padding (if last chunk is smaller than available seats)
        if len(chunk) < carriers_per_symbol:
            padding_needed = carriers_per_symbol - len(chunk)
            chunk = np.concatenate([chunk, np.zeros(padding_needed, dtype=complex)])
            
        symbol_freq[data_indices] = chunk
        #symbol_freq[N_FFT//2] = 0
        # IFFT (Frequency -> Time)
        # Standard numpy IFFT expects: [0, 1, ... N/2 ... -1]
        symbol_time = np.fft.ifft(symbol_freq)
        
        # Add Cyclic Prefix
        symbol_with_cp = np.concatenate([symbol_time[-N_CP:], symbol_time])
        payload_waves.append(symbol_with_cp)
        
    # Final Assembly
    # Structure: [Preamble] + [OFDM Symbol 1] + [OFDM Symbol 2] ...
    if len(payload_waves) > 0:
        full_payload = np.concatenate(payload_waves)
        tx_buffer = np.concatenate([pre_wave, full_payload])
    else:
        # Edge case: No data, just preamble
        tx_buffer = pre_wave

    # Hardware Scaling
    # Normalize to max amplitude 0.5 to prevent PlutoSDR clipping
    max_amp = np.max(np.abs(tx_buffer))
    if max_amp > 0:
        tx_buffer = (tx_buffer / max_amp) * 16384
        
    return tx_buffer