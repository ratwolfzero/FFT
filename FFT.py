from scipy.fft import fft, ifft
import warnings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# Works fine with MacOS w/o explicit backend. However, crashes when trying to manually resize the created plot window.
matplotlib.use('TkAgg')
warnings.filterwarnings("ignore")
plt.style.use('bmh')

# signal information
freq = 1      # in Hertz
cycle = 2      # number of cycles

# sampling information
Fs = int(freq*100)  # sample rate
T = 1/Fs           # sampling period
t = 1/freq*cycle   # seconds of sampling
N = Fs*t           # total points in signal

# time vector
t_vec = np.arange(N)*T

DC = 0

y0 = 1*np.sin(2*np.pi*freq*t_vec)
y1 = 1/3*np.sin(2*np.pi*3*freq*t_vec)
y2 = 1/5*np.sin(2*np.pi*5*freq*t_vec)
y3 = 1/7*np.sin(2*np.pi*7*freq*t_vec)
y4 = 1/9*np.sin(2*np.pi*9*freq*t_vec)
Y = DC+y0+y1+y2+y3+y4

# time domain
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.tight_layout(pad=4)

ax[0, 0].set_title('signals')
ax[0, 0].set_ylabel('Amplitude', fontsize=10)
ax[0, 0].set_xlabel('time [s]', fontsize=10)
ax[0, 0].plot(t_vec, y0)
ax[0, 0].plot(t_vec, y1)
ax[0, 0].plot(t_vec, y2)
ax[0, 0].plot(t_vec, y3)
ax[0, 0].plot(t_vec, y4)

ax[0, 1].set_title('resulting signal')
ax[0, 1].set_ylabel('amplitude', fontsize=10)
ax[0, 1].set_xlabel('time [s]', fontsize=10)
ax[0, 1].plot(t_vec, Y)

# Fast Fourier Transform (FFT)
Y_k = fft(Y)                      # scipy.fft
# single sided-sided spectrum only, normalize amplitude
Y_k_mag = Y_k[0:int(N/2)]/N
Y_k_mag[1:] = 2*Y_k_mag[1:]       # keep DC amplitude, correct AC amplitude
Pxx = np.abs(Y_k_mag)             # skip imaginary part

# frequency vector
f_vec = Fs*np.arange((N/2))/N

# frequency domain

#ax[1,0].set_xscale('symlog')
#ax[1, 0].set_yscale('log')
ax[1, 0].set_title('FFT')
ax[1, 0].set_ylabel('Amplitude', fontsize=10)
ax[1, 0].set_xlabel('frequncy [Hz]', fontsize=10)
ax[1,0].stem(f_vec,Pxx, 'r',markerfmt=" ", basefmt="-r")
#ax[1, 0].plot(f_vec, Pxx, 'r')

# time domain inverse FFT
Y_inv = ifft(Y_k)

ax[1, 1].set_title('iFFT')
ax[1, 1].set_ylabel('Amplitude', fontsize=10)
ax[1, 1].set_xlabel('time [s]', fontsize=10)
ax[1, 1].plot(t_vec, Y_inv, 'r')

plt.show()
