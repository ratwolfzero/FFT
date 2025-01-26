import numpy as np
import wave
import matplotlib.pyplot as plt

def generate_composite_signal_with_harmonics(duration, sampling_rate, fundamental_frequency, harmonics, amplitudes, dc_offset):
    """
    Generate a composite signal with selectable harmonics and a DC component.

    Args:
        duration (float): Duration of the signal in seconds.
        sampling_rate (int): Sampling rate in Hz.
        fundamental_frequency (float): Fundamental frequency in Hz.
        harmonics (list): List of harmonic numbers to include (e.g., [1, 3, 5] for 1st, 3rd, and 5th harmonics).
        amplitudes (list): List of amplitudes for each harmonic.
        dc_offset (float): DC offset to add to the signal.

    Returns:
        np.ndarray: The generated composite signal.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    composite_signal = dc_offset

    for harmonic, amplitude in zip(harmonics, amplitudes):
        composite_signal += amplitude * np.sin(2 * np.pi * harmonic * fundamental_frequency * t)

    return composite_signal

def save_signal_to_wav(filename, signal, sampling_rate):
    """
    Save a signal to a WAV file.

    Args:
        filename (str): Path to save the WAV file.
        signal (np.ndarray): Signal to save.
        sampling_rate (int): Sampling rate in Hz.
    """
    # Ensure the signal is within the valid range for int16 WAV files
    signal = np.clip(signal, -1.0, 1.0)

    # Scale to int16 range and save as WAV
    int_signal = (signal * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16 bits per sample
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(int_signal.tobytes())

def plot_composite_signal(t, signal, fundamental_frequency, title):
    """
    Plot the generated composite signal, showing only 10 cycles.

    Args:
        t (np.ndarray): Time vector.
        signal (np.ndarray): Signal to plot.
        fundamental_frequency (float): Fundamental frequency in Hz.
        title (str): Plot title.
    """
    num_samples_per_cycle = int(len(t) / (t[-1] * fundamental_frequency))
    num_samples_to_plot = num_samples_per_cycle * 10  # 10 cycles

    plt.figure(figsize=(10, 6))
    plt.plot(t[:num_samples_to_plot], signal[:num_samples_to_plot], label="Composite Signal")
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_fft_of_signal(signal, sampling_rate):
    """
    Plot the FFT of the generated signal to inspect its frequency components.

    Args:
        signal (np.ndarray): Signal to analyze.
        sampling_rate (int): Sampling rate in Hz.
    """
    num_samples = len(signal)
    signal_fft = np.fft.fft(signal)
    magnitude_spectrum = np.abs(signal_fft) / num_samples  # Normalize
    magnitude_spectrum[1:] *= 2  # Scale non-DC components
    frequency_vector = np.fft.fftfreq(num_samples, d=1/sampling_rate)

    # Plot FFT
    plt.figure(figsize=(10, 6))
    plt.plot(frequency_vector[:num_samples//2], magnitude_spectrum[:num_samples//2])
    plt.xscale('symlog')  # SymLog scale for frequency with DC
    plt.yscale('log')  # Log scale for magnitude
    plt.title("FFT of Composite Signal")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.show()

def main():
    # Parameters
    output_file = "signal_with_dc.wav"
    duration = 5.0  # 5 seconds
    sampling_rate = 44100  # 44.1 kHz
    fundamental_frequency = 100  
    harmonics = [1, 3, 5]  # Fundamental and harmonics
    amplitudes = [0.5, 0.3, 0.2]  # Amplitudes for each harmonic
    dc_offset = 0.2  # DC offset

    # Generate composite signal
    signal = generate_composite_signal_with_harmonics(
        duration, sampling_rate, fundamental_frequency, harmonics, amplitudes, dc_offset
    )

    # Save signal to WAV file
    save_signal_to_wav(output_file, signal, sampling_rate)
    print(f"File '{output_file}' generated with composite harmonics and DC component!")

    # Plot the signal
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    plot_composite_signal(t, signal, fundamental_frequency, "Generated Composite Signal with DC Component (10 Cycles)")

    # Plot the FFT of the signal
    plot_fft_of_signal(signal, sampling_rate)

if __name__ == "__main__":
    main()

