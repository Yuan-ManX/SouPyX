# MIT License

# Copyright (c) 2023 Yuan-Man

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import stft
import matplotlib.pyplot as plt


#------------------ Waveform ----------------------------------------------------

def file_waveform(filename):
    """
    Plot the waveform of an audio file.

    Args:
        filename: the name of the audio file.

    Returns:
        None.
    """
    # Read the audio file and get the sampling rate and audio data
    rate, data = wav.read(filename)

    # Convert the audio data to floating-point numbers
    data = data / 32768.0

    # Calculate the duration of the audio and the time axis
    duration = len(data) / rate
    time = np.linspace(0, duration, len(data))

    # Plot the waveform
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform of {}'.format(filename))
    plt.show()


def waveform(data, rate=44100):
    """
    Plot the waveform of an audio signal.

    Args:
        data: a 1D numpy array that contains the audio signal.
        rate: the sampling rate of the audio signal.

    Returns:
        None.
    """
    # Convert the audio data to floating-point numbers
    data = data / np.max(np.abs(data))

    # Calculate the duration of the audio and the time axis
    duration = len(data) / rate
    time = np.linspace(0, duration, len(data))

    # Plot the waveform
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()


#------------------ Spectrum ----------------------------------------------------

def file_spectrum(filename):
    """
    Plot the spectrum of an audio file.

    Args:
        filename: the name of the audio file.

    Returns:
        None.
    """
    # Read the audio file and get the sampling rate and audio data
    rate, data = wav.read(filename)

    # Compute the power spectrum of the audio signal
    fft_size = 2 ** int(np.ceil(np.log2(len(data))))
    spectrum = np.abs(np.fft.fft(data, fft_size)) ** 2
    frequency = np.fft.fftfreq(fft_size, 1 / rate)
    mask = frequency >= 0

    # Plot the spectrum
    plt.plot(frequency[mask], 10 * np.log10(spectrum[mask]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Spectrum of {}'.format(filename))
    plt.show()


def spectrum(data, rate=44100):
    """
    Plot the spectrum of an audio signal.

    Args:
        data: a numpy array containing the audio data.
        rate: the sampling rate of the audio data.

    Returns:
        None.
    """
    # Compute the power spectrum of the audio signal
    fft_size = 2 ** int(np.ceil(np.log2(len(data))))
    spectrum = np.abs(np.fft.fft(data, fft_size)) ** 2
    frequency = np.fft.fftfreq(fft_size, 1 / rate)
    mask = frequency >= 0

    # Plot the spectrum
    plt.plot(frequency[mask], 10 * np.log10(spectrum[mask]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Spectrum')
    plt.show()


#------------------ Spectrogram -------------------------------------------------

def file_spectrogram(filename, window='hann', nperseg=512, noverlap=256, cmap='viridis'):
    """
    Plot spectrogram of an audio file.

    Args:
        filename: str, the name of the audio file.
        window: str or tuple, optional. Type of window ('hann', 'hamming', 'rectangular', etc.) or the window itself.
                Defaults to 'hann'.
        nperseg: int, optional. Length of each segment. Defaults to 512.
        noverlap: int, optional. Number of samples to overlap between segments. Defaults to 256.
        cmap: str or colormap, optional. Colormap to use for pcolormesh. Defaults to 'viridis'.

    Returns:
        None
    """
    # Read audio file and get the sampling rate and data
    rate, data = wav.read(filename)

    # Convert the audio data to float
    data = data / 32768.0

    # Calculate the duration and time axis of the audio
    duration = len(data) / rate
    time = np.linspace(0, duration, len(data))

    # Calculate spectrogram using short-time Fourier transform (STFT)
    f, t, Zxx = stft(data, fs=rate, window=window, nperseg=nperseg, noverlap=noverlap)

    # Plot spectrogram
    plt.pcolormesh(t, f, np.abs(Zxx), cmap=cmap)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram of {}'.format(filename))
    plt.show()


def spectrogram(data, rate=44100, window_size=1024, step_size=256, eps=1e-10):
    """
    Plot the spectrogram of audio signal.

    Args:
        data: numpy array containing audio data
        rate: integer representing the sampling rate of audio data
        window_size: integer representing the window size in number of samples, default is 1024
        step_size: integer representing the step size in number of samples, default is 256
        eps: float controlling the value correction to avoid log(0), default is 1e-10

    Returns:
        None
    """
    # Compute FFT for each window
    window = np.hamming(window_size)
    hop_length = window_size // 2
    spec = np.zeros((window_size, 1 + len(data) // hop_length))
    for i, j in enumerate(range(0, len(data) - window_size, step_size)):
        spec[:, i] = np.abs(np.fft.rfft(window * data[j:j + window_size])) ** 2

    # Convert FFT to dB
    spec = 10 * np.log10(spec + eps)

    # Plot the spectrogram
    frequency = np.fft.rfftfreq(window_size, 1 / rate)
    time = np.arange(spec.shape[1]) * step_size / rate
    plt.pcolormesh(time, frequency, spec, cmap='viridis')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.colorbar().set_label('Power (dB)')
    plt.show()


#------------------ Waterfall --------------------------------------------------------

def file_waterfall(filename, window='hann', nperseg=512, noverlap=256, cmap='viridis', zmin=None, zmax=None):
    """
    Plot a waterfall visualization of an audio file.

    Args:
        filename (str): The filename of the audio file.
        window (str): The type of window to use. Default is 'hann'.
        nperseg (int): The length of each segment. Default is 512.
        noverlap (int): The amount of overlap between adjacent segments. Default is 256.
        cmap (str): The colormap to use. Default is 'viridis'.
        zmin (float): The minimum value for the Z-axis. Optional.
        zmax (float): The maximum value for the Z-axis. Optional.

    Returns:
        None
    """

    # Read the audio file and get the sample rate and data
    rate, data = wav.read(filename)

    # Convert the data to floating-point values
    data = data / 32768.0

    # Calculate the duration of the audio and the time axis
    duration = len(data) / rate
    time = np.linspace(0, duration, len(data))

    # Calculate the spectrogram
    f, t, Zxx = stft(data, fs=rate, window=window, nperseg=nperseg, noverlap=noverlap)

    # Plot the waterfall visualization
    fig, ax = plt.subplots()
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title('Waterfall of {}'.format(filename))
    waterfall = ax.pcolormesh(t, f, np.abs(Zxx), cmap=cmap, shading='nearest')
    ax.set_ylim([0, f[-1]])

    # Adjust the Z-axis range if zmin and zmax are specified
    if zmin is not None and zmax is not None:
        waterfall.set_clim(zmin, zmax)

    # Add a colorbar
    fig.colorbar(waterfall)

    plt.show()


def waterfall(data, fs=44100, nperseg=512, noverlap=256, cmap='viridis'):
    """
    Plot the waterfall plot of audio signal.

    Args:
        data: Audio data
        fs: Sampling rate
        nperseg: Length of each segment, default to 512
        noverlap: Number of points to overlap between segments, default to 256
        cmap: Colormap for the plot, default to 'viridis'

    Returns:
        None
    """
    # Calculate the spectrogram
    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Plot the waterfall plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(Zxx.shape[1]):
        ax.plot(t, f, np.abs(Zxx[:, i]), color=plt.cm.jet(i / Zxx.shape[1]))

    # Set the labels of the axes
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_zlabel('Amplitude')

    plt.show()


#------------------ 3D Spectrum --------------------------------------------------------

def file_spectrum_3d(filename, window='hann', nperseg=512, noverlap=256, cmap='viridis'):
    """
    Plot the 3D spectrum of an audio file.

    Args:
        filename (str): Name of the audio file.
        window (str): Type of window. Default is 'hann'.
        nperseg (int): Length of each segment. Default is 512.
        noverlap (int): Length of overlap between segments. Default is 256.
        cmap (str): Colormap. Default is 'viridis'.

    Returns:
        None
    """
    # Read audio file and get sample rate and audio data
    rate, data = wav.read(filename)

    # Convert audio data to float
    data = data / 32768.0

    # Compute the STFT
    f, t, Zxx = stft(data, fs=rate, window=window, nperseg=nperseg, noverlap=noverlap)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D spectrum
    X, Y = np.meshgrid(t, f)
    ax.plot_surface(X, Y, np.abs(Zxx), cmap=cmap)

    # Set axis labels and title
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_zlabel('Amplitude')
    ax.set_title('3D Spectrum of {}'.format(filename))

    plt.show()


def spectrum_3d(data, rate=44100, window='hann', nperseg=512, noverlap=256, cmap='viridis', zmin=None, zmax=None):
    """
    Plot 3D spectrum of audio signal.

    Args:
        data: Audio data as a numpy array.
        rate: Sampling rate of audio data.
        window: Type of window function. Default is 'hann'.
        nperseg: Length of each segment. Default is 512.
        noverlap: Number of points to overlap between segments. Default is 256.
        cmap: Color map to use for the plot. Default is 'viridis'.
        zmin: Minimum value of z-axis. Optional.
        zmax: Maximum value of z-axis. Optional.

    Returns:
        None
    """
    # Calculate duration and time axis of audio data.
    duration = len(data) / rate
    time = np.linspace(0, duration, len(data))

    # Calculate the spectrum.
    f, t, Zxx = stft(data, fs=rate, window=window, nperseg=nperseg, noverlap=noverlap)

    # Create 3D plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Generate mesh grid for x-axis and y-axis.
    T, F = np.meshgrid(t, f)

    # Plot 3D spectrum.
    surf = ax.plot_surface(T, F, np.abs(Zxx), cmap=cmap)

    # Set labels for x-axis and y-axis.
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Frequency [Hz]')

    # Adjust z-axis range.
    if zmin is not None and zmax is not None:
        surf.set_clim(zmin, zmax)

    # Add color bar.
    fig.colorbar(surf)

    plt.show()

