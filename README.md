# Real-Time-OFDM-Implementation-on-ADI-CN0566
This repository provides a real-time OFDM transceiver implementation. Built with Python, it leverages the ADALM-PLUTO Software-Defined Radio and the Analog Devices CN0566 Phaser kit. The system features 16-QAM modulation, adaptive timing synchronization, zero-forcing equalization, and live constellation visualization.

The project demonstrates a full end-to-end transceiver pipeline, including hardware calibration, custom beamforming weight application, 16-QAM symbol mapping, packet synchronization, channel estimation, and real-time data decoding.

## 🛠 Hardware Requirements

* **SDR:** ADALM-PLUTO (PlutoSDR) configured at `192.168.2.1`
* **Beamformer:** Analog Devices CN0566 Phaser Kit configured at `phaser.local`
* **Optional/Test Source:** HB100 Microwave Sensor (used for frequency calibration/interference tracking)

## 📦 Software Dependencies

Ensure you have the following Python libraries installed:
* `numpy`
* `scipy`
* `matplotlib`
* `pyadi-iio` (imported as `adi`)
* `cupy` (for GPU-accelerated processing, optional depending on your environment)

## 🗂 Project Structure

* **`config.py`**: Central configuration file defining IP addresses, sampling rates (2.5 Msps), Tx/Rx frequencies (2.2 GHz), buffer sizes, and manual calibration offsets.
* **`calibration.py`**: Contains routines to locate the HB100 peak frequency, perform channel gain/phase calibration across the Phaser's 8 elements, and execute beamsteering sweeps to plot angle-of-arrival.
* **`helpers.py`**: The core DSP engine for the OFDM system. It handles string-to-bit conversion, 16-QAM constellation mapping, Schmidl & Cox preamble generation, pilot insertion, and time-domain OFDM buffer assembly (including Cyclic Prefix addition).

* **`OFDM_main.py`**: A single-shot script that configures the SDR and Phaser, applies specific pre-calculated beamforming weights, transmits a user-defined string as an OFDM payload, and decodes the received buffer. It features an adaptive timing scan to perfectly align the synchronization preamble.
* **`realtime_OFDM.py`**: A continuous-loop version of the OFDM system. It transmits a cyclic buffer and uses cross-correlation to synchronize, equalize, and decode incoming frames in real-time, outputting dynamic constellation and frequency domain plots.

## 🚀 How It Works

### 1. Modulation & Framing
Text data is converted to bits and mapped to a 16-QAM constellation. 

The system uses a 128-point FFT with dedicated subcarriers for data, pilots (for channel estimation), and guard bands (to prevent aliasing). A cyclic prefix of 32 samples is prepended to each symbol to mitigate inter-symbol interference (ISI).

### 2. Synchronization
The transmitter sends a known preamble before the data payload. The receiver uses cross-correlation to find this preamble in the time domain, determining the exact starting index of the OFDM frame. 

### 3. Equalization
Because the wireless channel distorts the signal, the receiver extracts the known pilot symbols and compares them to what was transmitted. Using linear interpolation, it estimates the channel response across all subcarriers. Equalization is performed using Zero-Forcing, represented as:
$$\hat{X} = \frac{Y}{\hat{H}}$$
where $Y$ is the received symbol, $\hat{H}$ is the estimated channel, and $\hat{X}$ is the recovered data.

## ⚙️ Usage

**1. Hardware Setup:** Ensure your PlutoSDR and Raspberry Pi (Phaser) are powered and connected to your network. 

**2. Calibration (Optional but recommended):**
Run the calibration script to measure the interference frequency, align the phased array elements, and save the calibration weights:
```bash
python calibration.py
```

**3. Single-Shot Transmission:**
To transmit a single message, apply beamforming weights, and view the static constellation plot:
```bash
python OFDM_main.py
```
*You will be prompted to enter a data string in the console.*

**4. Real-Time Processing:**
To launch the continuous transceiver loop with live Matplotlib visualization:
```bash
python realtime_OFDM.py
```
*Press Ctrl+C to stop the real-time reception and safely destroy the Tx buffer.*

https://github.com/user-attachments/assets/771e2543-48f7-4dfc-af04-1c4f0cde2a68


