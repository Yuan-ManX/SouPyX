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
from scipy.signal import lfilter, butter
from scipy import signal


#------------------ Filter ----------------------------------------------------

def filter(audio_data, fs, filter_type='lowpass', cutoff_freq=1000):
    """
    Applies a digital filter to the input audio signal.

    Args:
        audio_data (numpy array): Input audio signal.
        fs (int): Sampling rate of the audio signal.
        filter_type (str): Type of filter to apply ('lowpass', 'highpass', 'bandpass', or 'bandstop').
        cutoff_freq (int or tuple): Cutoff frequency or frequencies (depending on filter type).
        
    Returns:
        numpy array: Filtered audio signal.
    """
    nyquist_freq = fs / 2

    if filter_type == 'lowpass':
        b, a = butter(4, cutoff_freq / nyquist_freq, 'low')
    elif filter_type == 'highpass':
        b, a = butter(4, cutoff_freq / nyquist_freq, 'high')
    elif filter_type == 'bandpass':
        b, a = butter(4, [cutoff_freq[0] / nyquist_freq, cutoff_freq[1] / nyquist_freq], 'bandpass')
    elif filter_type == 'bandstop':
        b, a = butter(4, [cutoff_freq[0] / nyquist_freq, cutoff_freq[1] / nyquist_freq], 'bandstop')

    filtered_signal = signal.lfilter(b, a, audio_data)

    return filtered_signal


#------------------ Compressor -------------------------------------------------

def compressor(input_signal, threshold, ratio, attack_time, release_time, sample_rate):
    """
    Applies a compressor effect to the input audio signal.

    Parameters:
        input_signal (ndarray): Input audio signal
        threshold (float): Amplitude threshold for compression
        ratio (float): Compression ratio
        attack_time (float): Time taken for the compressor to reach full compression level
        release_time (float): Time taken for the compressor to release full compression level
        sample_rate (int): Sampling rate of the audio signal

    Returns:
        ndarray: Output audio signal
    """

    # Compute the envelope of the input signal
    envelope = np.abs(input_signal)
    # Compute the gain reduction
    gain_reduction = np.zeros_like(envelope)
    gain_reduction[envelope > threshold] = (threshold + (envelope[envelope > threshold] - threshold) / ratio) / envelope[envelope > threshold]
    # Apply attack and release to the gain reduction
    attack_frames = int(attack_time * sample_rate)
    release_frames = int(release_time * sample_rate)
    gain_reduction = lfilter([1], np.concatenate(([1] * attack_frames, [0] * (len(gain_reduction) - attack_frames - release_frames), [1] * release_frames)), gain_reduction)

    # Apply gain reduction to the input signal
    output_signal = input_signal * gain_reduction

    return output_signal


#------------------ Delayer -------------------------------------------------

def delayer(signal, delay_time=500, feedback=0.5, mix=0.5):
    """
    Applies a delay effect to the input audio signal.

    Args:
        signal (numpy array): Input audio signal.
        delay_time (float): Delay time (in ms).
        feedback (float): Feedback gain (between 0 and 1).
        mix (float): Dry/wet mix (between 0 and 1).
        
    Returns:
        numpy array: Delayed audio signal.
    """
    delay_samples = int(delay_time / 1000 * fs)
    output_signal = np.zeros_like(signal)
    delayed_signal = np.zeros_like(signal)

    for i in range(delay_samples, len(signal)):
        delayed_signal[i] = signal[i - delay_samples] + feedback * delayed_signal[i - delay_samples]

    output_signal = signal * (1 - mix) + delayed_signal * mix

    return output_signal


#------------------ Sampler -------------------------------------------------

def sampler(signal, sample_rate_ratio=0.5):
    """
    Applies a sample rate reducer effect to the input audio signal.

    Args:
        signal (numpy array): Input audio signal.
        sample_rate_ratio (float): Sampling rate reduction ratio.
        
    Returns:
        numpy array: Sample rate reduced audio signal.
    """
    output_signal = np.zeros_like(signal)
    step_size = int(1 / sample_rate_ratio)
    output_signal[::step_size] = signal[::step_size]

    return output_signal


#------------------ Equalize -------------------------------------------------

def equalizer(input_signal, center_freq, gain, Q, sample_rate):
    """
    Applies an equalizer effect to the input audio signal.

    Parameters:
        input_signal (ndarray): Input audio signal
        center_freq (float): Center frequency of the equalizer band
        gain (float): Amount of gain to apply to the band
        Q (float): Q factor of the equalizer band
        sample_rate (int): Sampling rate of the audio signal

    Returns:
        ndarray: Output audio signal
    """

    # Compute filter coefficients
    w0 = 2 * np.pi * center_freq / sample_rate
    alpha = np.sin(w0) / (2 * Q)
    A = np.sqrt(10 ** (gain / 20))
    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    # Apply filter to the input signal
    output_signal = lfilter([b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0], input_signal)

    return output_signal


#------------------ Doppler -------------------------------------------------

def doppler(signal, fs, speed=1, doppler_shift=1):
    """
    Applies a Doppler shift effect to the input audio signal.

    Args:
        signal (numpy array): Input audio signal.
        fs (int): Sampling rate of the audio signal.
        speed (float): Speed of the moving object.
        doppler_shift (float): Doppler shift amount (in Hz).
        
    Returns:
        numpy array: Doppler shifted audio signal.
    """
    c = 343  # Speed of sound (m/s)
    speed_of_source = speed  # Speed of source
    speed_of_observer = 0  # Speed of observer (microphone)

    # Calculate the Doppler shift frequency
    doppler_shift_frequency = (fs / c) * (doppler_shift / (doppler_shift + speed_of_source - speed_of_observer))

    # Create the Doppler shift effect
    t = np.arange(len(signal)) / fs
    phase_shift = 2 * np.pi * doppler_shift_frequency * t
    doppler_shifted_signal = signal * np.cos(phase_shift)

    return doppler_shifted_signal


#------------------ Pitch_Shift -------------------------------------------------

def pitch_shift(input_signal, pitch_shift_amount, sample_rate):
    """
    Applies a pitch shift effect to the input audio signal.

    Parameters:
    input_signal (ndarray): Input audio signal
    pitch_shift_amount (float): Amount of pitch shift to apply in semitones
    sample_rate (int): Sampling rate of the audio signal

    Returns:
        ndarray: Output audio signal
    """

    # Compute the FFT of the input signal
    input_signal_fft = np.fft.rfft(input_signal)

    # Compute the frequency bins of the FFT
    freq_bins = np.fft.rfftfreq(len(input_signal), 1/sample_rate)

    # Compute the amount of phase shift to apply to each frequency bin
    phase_shift = 2 * np.pi * pitch_shift_amount * freq_bins / 12

    # Shift the phase of the FFT
    shifted_signal_fft = input_signal_fft * np.exp(1j * phase_shift)

    # Compute the inverse FFT of the shifted signal
    output_signal = np.fft.irfft(shifted_signal_fft)

    return output_signal


#------------------ Flanger -------------------------------------------------

def flanger(signal, delay_time, depth, rate, fs):
    """
    Apply a flanger effect to an audio signal.

    Args:
        signal (numpy array): Input audio signal.
        delay_time (float): Delay time (in seconds) of the flanger effect.
        depth (float): Depth of the flanger effect (in seconds).
        rate (float): LFO rate (in Hz) of the flanger effect.
        fs (int): Sampling rate of the audio signal.

    Returns:
        numpy array: Flanged audio signal.
    """
    # Create LFO signal
    t = np.arange(len(signal)) / fs
    lfo_signal = (1 + depth * np.sin(2 * np.pi * rate * t)) / 2

    # Create delayed signal
    delay_samples = int(round(delay_time * fs))
    delay_signal = np.zeros(len(signal) + delay_samples)
    delay_signal[delay_samples:] = signal

    # Apply flanging effect
    flanged_signal = signal + lfo_signal * delay_signal[:len(signal)]

    return flanged_signal


#------------------ Chorus -------------------------------------------------

def chorus(signal, depth, rate, width, fs):
    """
    Apply a chorus effect to an audio signal.

    Args:
        signal (numpy array): Input audio signal.
        depth (float): Depth of the chorus effect (in seconds).
        rate (float): LFO rate (in Hz) of the chorus effect.
        width (float): Width of the stereo image (in seconds).
        fs (int): Sampling rate of the audio signal.

    Returns:
        numpy array: Chorused audio signal.
    """
    # Create LFO signal
    t = np.arange(len(signal)) / fs
    lfo_signal = depth * np.sin(2 * np.pi * rate * t)

    # Create delayed signals
    width_samples = int(round(width * fs))
    left_delay_signal = np.zeros(len(signal) + width_samples)
    right_delay_signal = np.zeros(len(signal) + width_samples)
    left_delay_signal[width_samples:] = signal
    right_delay_signal[width_samples:] = signal

    # Apply chorus effect
    left_chorus = signal + lfo_signal * left_delay_signal[:len(signal)]
    right_chorus = signal + lfo_signal * right_delay_signal[:len(signal)]

    # Mix stereo image
    chorus_signal = np.vstack((left_chorus, right_chorus)).T
    return chorus_signal


#------------------ Reverb ----------------------------------------------------

def reverb(signal, ir, fs):
    """
    Apply a reverb effect to an audio signal using a given impulse response.

    Args:
        signal (numpy array): Input audio signal.
        ir (numpy array): Impulse response of the reverb.
        fs (int): Sampling rate of the audio signal and impulse response.

    Returns:
        numpy array: Reverberant audio signal.
    """
    # Normalize impulse response
    ir = ir / np.max(ir)

    # Pad signal and impulse response with zeros
    max_len = len(signal) + len(ir)
    signal = np.pad(signal, (0, max_len - len(signal)))
    ir = np.pad(ir, (0, max_len - len(ir)))

    # Compute FFT of signal and impulse response
    signal_fft = np.fft.rfft(signal)
    ir_fft = np.fft.rfft(ir)

    # Multiply FFT of signal and impulse response
    reverberant_fft = signal_fft * ir_fft

    # Compute inverse FFT of reverberant signal
    reverberant_signal = np.fft.irfft(reverberant_fft)

    return reverberant_signal


#------------------ Modulation -------------------------------------------------

def modulation(signal, mod_signal, depth):
    """
    Apply a modulation effect to an audio signal.

    Args:
        signal (numpy array): Input audio signal.
        mod_signal (numpy array): Modulation signal.
        depth (float): Depth of the modulation effect.

    Returns:
        numpy array: Modulated audio signal.
    """
    # Create modulation envelope
    mod_env = 1 + depth * mod_signal

    # Apply modulation effect
    modulated_signal = signal * mod_env

    return modulated_signal


