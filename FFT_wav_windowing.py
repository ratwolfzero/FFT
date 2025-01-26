import os
import wave
import numpy as np
from scipy.fft import fft, ifft
from scipy.signal.windows import get_window
import matplotlib.pyplot as plt
import sounddevice as sd


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
    extracted_signal -= np.mean(extracted_signal)
    time_vector = np.arange(samples_to_extract) / sampling_rate

    return extracted_signal, time_vector


def apply_window(time_domain_signal, window_type='hann'):
    # Apply the selected window (default is Hann window) to the signal. Options e.g. blackman, flat.
    window = get_window(window_type, len(time_domain_signal))
    return time_domain_signal * window


def compute_fft(time_domain_signal, sampling_rate):
    """
    Computes the Fast Fourier Transform (FFT) of a time-domain signal and returns 
    the frequency vector, magnitude spectrum, and the FFT result.

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
    
    return frequency_vector, magnitude_spectrum, fft_result


def reconstruct_signal_from_fft(fft_result):
    return ifft(fft_result).real


def plot_results(windowed_signal, time_vector, frequency_vector, magnitude_spectrum, reconstructed_signal, sampling_rate):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.tight_layout(pad=4)

    ax[0, 0].set_title('Windowed Signal (Extracted Segment)')
    ax[0, 0].plot(time_vector, windowed_signal, color='b')
    ax[0, 0].set_xlabel('Time [s]')
    ax[0, 0].set_ylabel('Amplitude')
    ax[0, 0].grid()

    ax[0, 1].set_title('Spectrogram of Windowed Signal')
    ax[0, 1].specgram(windowed_signal, Fs=sampling_rate, NFFT=256, noverlap=128, cmap='viridis')
    ax[0, 1].set_xlabel('Time [s]')
    ax[0, 1].set_ylabel('Frequency [Hz]')

    ax[1, 0].set_title('FFT (Frequency Domain)')
    ax[1, 0].stem(frequency_vector, magnitude_spectrum, linefmt='r-', markerfmt=' ', basefmt='k-')
    ax[1, 0].set_xlabel('Frequency [Hz]')
    ax[1, 0].set_ylabel('Magnitude')
    ax[1, 0].set_xscale('log')
    ax[1, 0].grid()

    ax[1, 1].set_title('Reconstructed Signal (iFFT)')
    ax[1, 1].plot(time_vector, reconstructed_signal[:len(time_vector)], color='g')
    ax[1, 1].set_xlabel('Time [s]')
    ax[1, 1].set_ylabel('Amplitude')
    ax[1, 1].grid()

    plt.show()


def main():
    file_path = os.path.expanduser("~/Projects/Python/FFT/signal1.wav")
    time_domain_signal, sampling_rate = load_wav_file(file_path)
    frequency = 100
    """
    Extract a higher number of cycles to reduce the symmetric spectral spreading caused by windowing,
    as windowing introduces side-lobe energy around the fundamental and harmonic frequencies.
    """
    num_cycles = 250

    extracted_signal, time_vector = extract_cycles(time_domain_signal, sampling_rate, frequency, num_cycles)
    windowed_signal = apply_window(extracted_signal)

    # Play sound first, then process
    sd.play(windowed_signal, sampling_rate)
    sd.wait()

    frequency_vector, magnitude_spectrum, fft_result = compute_fft(windowed_signal, sampling_rate)
    reconstructed_signal = reconstruct_signal_from_fft(fft_result)

    plot_results(windowed_signal, time_vector, frequency_vector, magnitude_spectrum, reconstructed_signal, sampling_rate)

if __name__ == "__main__":
    main()


