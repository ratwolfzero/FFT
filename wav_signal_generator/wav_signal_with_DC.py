import numpy as np
import wave
import matplotlib.pyplot as plt

def generate_wav_with_dc(filename, duration, sampling_rate, amplitude, dc_offset):
    """
    Generate a WAV file with a DC component.

    Args:
        filename (str): Path to save the WAV file.
        duration (float): Duration of the signal in seconds.
        sampling_rate (int): Sampling rate in Hz.
        amplitude (float): Amplitude of the sine wave (range: -1 to 1).
        dc_offset (float): DC offset to add to the signal (range: -1 to 1).
    """
    # Generate a sine wave
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave

    # Add DC offset
    signal = sine_wave + dc_offset

    # Ensure the signal is within the valid range for int16 WAV files
    signal = np.clip(signal, -1.0, 1.0)

    # Scale to int16 range and save as WAV
    int_signal = (signal * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16 bits per sample
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(int_signal.tobytes())

def plot_signal_with_dc(duration, sampling_rate, amplitude, dc_offset):
    """
    Plot the generated signal to visually inspect the DC offset and sine wave.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * 1 * t)
    signal = sine_wave + dc_offset

    # Plot the signal
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label="Signal with DC offset")
    plt.title("Generated Signal with DC Component")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_fft_of_signal(duration, sampling_rate, amplitude, dc_offset):
    """
    Plot the FFT of the generated signal to inspect the DC component in the frequency domain.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * 100 * t)
    signal = sine_wave + dc_offset

    # Perform FFT
    signal_fft = np.fft.fft(signal)
    magnitude_spectrum = np.abs(signal_fft) / len(signal)  # Normalize
    frequency_vector = np.fft.fftfreq(len(signal), d=1/sampling_rate)

    # Plot FFT with log scale
    plt.figure(figsize=(10, 6))
    plt.plot(frequency_vector[:len(frequency_vector)//2], magnitude_spectrum[:len(magnitude_spectrum)//2])
    plt.xscale('symlog')  # SymLog scale for frequency with DC
    plt.yscale('log')  # Log scale for magnitude
    plt.title("FFT of Signal with DC Component")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude (log scale)")
    plt.grid(True)
    plt.show()

# Parameters
output_file = "signal+DC.wav"
duration = 5.0  # 5 seconds
sampling_rate = 44100  # 44.1 kHz
amplitude = 0.5  # Amplitude of sine wave
dc_offset = 0.2  # DC offset

# Generate and save the WAV file
generate_wav_with_dc(output_file, duration, sampling_rate, amplitude, dc_offset)
print(f"File '{output_file}' generated with a DC component!")

# Plot the generated signal with DC component
plot_signal_with_dc(duration, sampling_rate, amplitude, dc_offset)

# Plot the FFT of the generated signal to inspect the DC component
plot_fft_of_signal(duration, sampling_rate, amplitude, dc_offset)