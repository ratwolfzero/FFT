import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft


# Function to generate individual harmonics and the composite signal
def generate_harmonics_and_composite(fundamental_frequency, time_vector, num_harmonics):
    """
    Generate individual odd harmonics (excluding the fundamental frequency) 
    and the composite signal.
    """
    signal_components = [
        (1 / harmonic) * np.sin(2 * np.pi * harmonic * fundamental_frequency * time_vector)
        for harmonic in range(3, 2 * num_harmonics + 3, 2)  # Start at the 2nd odd harmonic (frequency 3f)
    ]
    composite_signal = sum(signal_components)
    return signal_components, composite_signal


# Function to generate the time vector
def generate_time_vector(fundamental_frequency, num_cycles, sampling_rate):
    sampling_period = 1 / sampling_rate  # Sampling period
    total_duration = num_cycles / fundamental_frequency  # Total duration of sampling
    time_vector = np.arange(0, total_duration, sampling_period)  # Time vector
    return time_vector

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


# Function to compute inverse FFT
def compute_ifft(fft_result):
    return ifft(fft_result).real


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


# Function to plot the results
def plot_results(time_vector, signal_components, composite_signal, frequency_vector, magnitude_spectrum, reconstructed_signal, time_energy, freq_energy):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.tight_layout(pad=4)

    # Plot Fundamental frequency and individual harmonics
    ax[0, 0].set_title('Funsamantal Frequency & Harmonics')
    ax[0, 0].set_ylabel('Amplitude', fontsize=10)
    ax[0, 0].set_xlabel('Time [s]', fontsize=10)
    for component in signal_components:
        ax[0, 0].plot(time_vector, component)

    # Plot resulting composite signal
    ax[0, 1].set_title('Composite Signal (Time Domain)')
    ax[0, 1].set_ylabel('Amplitude', fontsize=10)
    ax[0, 1].set_xlabel('Time [s]', fontsize=10)
    ax[0, 1].plot(time_vector, composite_signal)

    # Plot FFT results
    ax[1, 0].set_title('FFT (Frequency Domain)')
    ax[1, 0].set_ylabel('Magnitude', fontsize=10)
    ax[1, 0].set_xlabel('Frequency [Hz]', fontsize=10)
    ax[1, 0].stem(frequency_vector, magnitude_spectrum, 'r', markerfmt=" ", basefmt="-r")
    ax[1, 0].set_xscale('log')

    ## Display energy values directly on the plot as text
    energy_text = (f"Energy (Time Domain): {time_energy:.4f}\n"
                   f"Energy (Frequency Domain): {freq_energy:.4f}")
    ax[1, 0].text(0.95, 0.95, energy_text, transform=ax[1, 0].transAxes, fontsize=10,
                  verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    # Plot iFFT (reconstructed signal)
    ax[1, 1].set_title('Inverse FFT (Reconstructed Composite Signal)')
    ax[1, 1].set_ylabel('Amplitude', fontsize=10)
    ax[1, 1].set_xlabel('Time [s]', fontsize=10)
    ax[1, 1].plot(time_vector, reconstructed_signal, 'r')

    plt.show()
    

# Main function to coordinate all steps
def main():
    fundamental_frequency = 100  # Fundamental frequency in Hz
    num_cycles = 8  # Number of cycles
    sampling_rate = 44100
    num_harmonics = 3  # Number of odd harmonics excluding the fundamental frequency

    # Generate time vector
    time_vector = generate_time_vector(fundamental_frequency, num_cycles, sampling_rate)

    # Generate individual harmonics and composite signal
    signal_components, composite_signal = generate_harmonics_and_composite(
        fundamental_frequency, time_vector, num_harmonics
    )

    # Add the fundamental frequency to the composite signal
    fundamental_component = np.sin(2 * np.pi * fundamental_frequency * time_vector)
    composite_signal += fundamental_component
    signal_components.insert(0, fundamental_component)  # Add it as the first component

    # Compute FFT
    frequency_vector, magnitude_spectrum, fft_result, num_samples = compute_fft(composite_signal, sampling_rate)

    # Compute inverse FFT (reconstruction)
    reconstructed_signal = compute_ifft(fft_result)

    time_energy, freq_energy = calculate_energy(composite_signal, magnitude_spectrum, num_samples)
    print(f"Energy in Time Domain: {time_energy:.4f}")
    print(f"Energy in Frequency Domain: {freq_energy:.4f}")

    # Plot results
    plot_results(time_vector, signal_components, composite_signal, frequency_vector, magnitude_spectrum, reconstructed_signal, time_energy, freq_energy)

if __name__ == "__main__":
    main()