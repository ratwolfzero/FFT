import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
import wave
import sounddevice as sd
import os


# Utility Functions
def load_wav_file(file_path):
    # Open the WAV file and read basic properties (sampling rate, frames, channels)
    with wave.open(file_path, 'rb') as wav_file:
        sampling_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        n_channels = wav_file.getnchannels()
        time_domain_signal = np.frombuffer(wav_file.readframes(n_frames), dtype=np.int16)
    
    # If stereo (multi-channel), average across channels to create a mono signal
    if n_channels > 1:
        time_domain_signal = time_domain_signal.reshape(-1, n_channels).mean(axis=1)

    # Normalize the signal to the range of [-1, 1] for 16-bit PCM data
    time_domain_signal = time_domain_signal / np.iinfo(np.int16).max

    return time_domain_signal, sampling_rate


def extract_cycles(time_domain_signal, sampling_rate, frequency, num_cycles):
    # Calculate the number of samples per cycle based on the frequency
    samples_per_cycle = int(sampling_rate / frequency)
    samples_to_extract = num_cycles * samples_per_cycle
    
    # Ensure the signal is long enough to extract the desired number of cycles
    if len(time_domain_signal) < samples_to_extract:
        raise ValueError("Signal too short for the requested number of cycles.")

    # Extract the portion of the signal for the requested number of cycles and remove DC offset
    extracted_signal = time_domain_signal[:samples_to_extract]
    extracted_signal -= np.mean(extracted_signal)  # Remove DC offset
    time_vector = np.arange(samples_to_extract) / sampling_rate

    return extracted_signal, time_vector


def compute_fft(time_domain_signal, sampling_rate):
    """
    Computes the Fast Fourier Transform (FFT) of a time-domain signal and returns 
    the frequency vector, magnitude spectrum, FFT result, and number of samples.

    Steps:
        1. Determine the number of samples in the input signal.
        2. Perform the FFT to convert the time-domain signal to the frequency domain.
        3. Extract the positive frequencies (first half of the FFT result).
        4. Compute the magnitude of the positive frequencies.
           - Normalize by the number of samples.
           - Scale non-DC components by 2 to account for the negative frequency contribution.
        5. Generate the frequency vector corresponding to the magnitude spectrum.
    """
    num_samples = len(time_domain_signal)
    fft_result = fft(time_domain_signal)
    positive_frequencies = fft_result[:num_samples // 2]
    magnitude_spectrum = np.abs(positive_frequencies) / num_samples  # Normalize by the number of samples
    magnitude_spectrum[1:] *= 2  # Scale non-DC components
    frequency_vector = sampling_rate * np.arange(num_samples // 2) / num_samples
    
    return frequency_vector, magnitude_spectrum, fft_result, num_samples


def calculate_energy(time_domain_signal, magnitude_spectrum, num_samples):
    """
    Computes the energy in both the time and frequency domains to verify signal consistency.

    - In the time domain, energy is the sum of squared amplitudes of the signal.
    - In the frequency domain, energy is the sum of squared magnitudes of the spectrum, 
      adjusted for FFT normalization.

    Since amplitudes/magnitudes in each domain can differ due to normalization and scaling, 
    this check ensures energy consistency between representations.
    """
    energy_time_domain = np.sum(time_domain_signal**2)  # Sum of squared signal values
    
    # Compute energy in the frequency domain (Parseval's theorem consideration)
    energy_frequency_domain = np.sum(magnitude_spectrum**2) * (num_samples // 2)

    return energy_time_domain, energy_frequency_domain


def reconstruct_signal_from_fft(fft_result):
    return ifft(fft_result).real


def plot_results(extracted_signal, time_vector, frequency_vector, magnitude_spectrum, reconstructed_signal, time_energy, freq_energy):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.tight_layout(pad=4)

    ax[0, 0].set_title('Original Signal (Extracted Segment)')
    ax[0, 0].plot(time_vector, extracted_signal, color='b')
    ax[0, 0].set_xlabel('Time [s]')
    ax[0, 0].set_ylabel('Amplitude')
    ax[0, 0].grid()

    ax[0, 1].set_title('FFT (Frequency Domain)')
    ax[0, 1].stem(frequency_vector, magnitude_spectrum, linefmt='r-', markerfmt=' ', basefmt='k-')
    ax[0, 1].set_xlabel('Frequency [Hz]')
    ax[0, 1].set_ylabel('Magnitude')
    ax[0, 1].set_xscale('log')
    ax[0, 1].grid()
    energy_text = (f"Energy (Time Domain): {time_energy:.4f}\n"
                   f"Energy (Frequency Domain): {freq_energy:.4f}")
    ax[0, 1].text(0.95, 0.95, energy_text, transform=ax[0, 1].transAxes, fontsize=10,
                  verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    power_spectrum = magnitude_spectrum**2
    ax[1, 0].plot(frequency_vector, power_spectrum)
    ax[1, 0].set_title('Power Spectrum')
    ax[1, 0].set_xlabel('Frequency [Hz]')
    ax[1, 0].set_ylabel('Power')
    ax[1, 0].set_xscale('log')
    ax[1, 0].grid()

    ax[1, 1].set_title('Reconstructed Signal (iFFT)')
    ax[1, 1].plot(time_vector, reconstructed_signal[:len(time_vector)], color='g')
    ax[1, 1].set_xlabel('Time [s]')
    ax[1, 1].set_ylabel('Amplitude')
    ax[1, 1].grid()

    plt.show()


def play_signal(time_domain_signal, sampling_rate):
    sd.play(time_domain_signal, sampling_rate)
    sd.wait()


def main():
    file_path = os.path.expanduser("~/Projects/Python/FFT/signal1.wav" )
    time_domain_signal, sampling_rate = load_wav_file(file_path)
    frequency = 100
    num_cycles = 8
    extracted_signal, time_vector = extract_cycles(time_domain_signal, sampling_rate, frequency, num_cycles)

# Play sound first, then process
    sd.play(extracted_signal, sampling_rate)
    sd.wait()

    frequency_vector, magnitude_spectrum, fft_result, num_samples = compute_fft(extracted_signal, sampling_rate)
    reconstructed_signal = reconstruct_signal_from_fft(fft_result)

    time_energy, freq_energy = calculate_energy(extracted_signal, magnitude_spectrum, num_samples)
    print(f"Energy in Time Domain: {time_energy:.4f}")
    print(f"Energy in Frequency Domain: {freq_energy:.4f}")

    plot_results(extracted_signal, time_vector, frequency_vector, magnitude_spectrum, reconstructed_signal, time_energy, freq_energy)

if __name__ == "__main__":
    main()



