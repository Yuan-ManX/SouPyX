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
from scipy import signal


#------------------ Sine Wave --------------------------------------------------
def sine_wave(frequency, duration, amplitude, sample_rate):
    """
    Generate a sine wave with the given frequency, duration, amplitude, and sample rate.

    Args:
    frequency (float): The frequency of the sine wave.
    duration (float): The duration of the sine wave in seconds.
    amplitude (float): The amplitude of the sine wave.
    sample_rate (int): The sample rate of the sine wave.

    Returns:
        numpy array: The generated sine wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


#------------------ Square Wave ------------------------------------------------

def square_wave(frequency, duration, amplitude, duty_cycle, sample_rate):
    """
    Generate a square wave with the given frequency, duration, amplitude, duty cycle, and sample rate.

    Args:
    frequency (float): The frequency of the square wave.
    duration (float): The duration of the square wave in seconds.
    amplitude (float): The amplitude of the square wave.
    duty_cycle (float): The duty cycle of the square wave.
    sample_rate (int): The sample rate of the square wave.

    Returns:
        numpy array: The generated square wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sign(np.sin(2 * np.pi * frequency * t) - duty_cycle + 0.5)


#------------------ Triangle Wave ----------------------------------------------

def triangle_wave(freq, duration, amplitude, fs):
    """
    Generates a triangle wave with the given frequency, duration, amplitude, and sampling rate.
    
    Args:
    freq (float): Frequency of the triangle wave (in Hz).
    duration (float): Duration of the triangle wave (in seconds).
    amplitude (float): Amplitude of the triangle wave.
    fs (int): Sampling rate of the triangle wave.

    Returns:
    numpy array: Triangle wave signal.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    triangle_wave = amplitude * signal.sawtooth(2 * np.pi * freq * t, width=0.5)

    return triangle_wave


#------------------ Sawtooth Wave ----------------------------------------------

def sawtooth_wave(freq, duration, amplitude, fs):
    """
    Generates a sawtooth wave with the given frequency, duration, amplitude, and sampling rate.
    
    Args:
    freq (float): Frequency of the sawtooth wave (in Hz).
    duration (float): Duration of the sawtooth wave (in seconds).
    amplitude (float): Amplitude of the sawtooth wave.
    fs (int): Sampling rate of the sawtooth wave.

    Returns:
    numpy array: Sawtooth wave signal.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sawtooth_wave = amplitude * signal.sawtooth(2 * np.pi * freq * t)

    return sawtooth_wave


#------------------ Noise Wave -------------------------------------------------

def noise_wave(duration, amplitude, sample_rate):
    """
    Generate a random noise wave signal with a given duration and amplitude.
    
    Args:
    duration (float): Duration of the generated signal (in seconds).
    amplitude (float): Amplitude of the generated signal.
    sample_rate (int): Sampling rate of the generated signal.

    Returns:
        numpy array: Generated noise wave signal.
    """

    # Compute the number of samples
    num_samples = int(duration * sample_rate)

    # Generate random noise samples
    noise_samples = amplitude * np.random.uniform(-1, 1, num_samples)

    return noise_samples


#------------------ Oscillator -------------------------------------------------

def oscillator(freq, duration, type='sine', sample_rate=44100):
    """
    Generates a waveform of the given frequency and duration.

    Args:
    freq (float): Frequency of the waveform (in Hz).
    duration (float): Duration of the waveform (in seconds).
    type (str): Type of waveform to generate (sine, square, sawtooth, triangle).
    sample_rate (int): Sampling rate of the waveform.

    Returns:
    numpy array: Generated waveform.
    """
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples, endpoint=False)
    if type == 'sine':
        waveform = np.sin(2 * np.pi * freq * t)
    elif type == 'square':
        waveform = np.sign(np.sin(2 * np.pi * freq * t))
    elif type == 'sawtooth':
        waveform = signal.sawtooth(2 * np.pi * freq * t)
    elif type == 'triangle':
        waveform = signal.sawtooth(2 * np.pi * freq * t, 0.5)
    else:
        raise ValueError("Invalid waveform type")

    return waveform


#------------------ ADSR -------------------------------------------------------

def adsr(audio_signal, fs=44100, attack_time=0.1, decay_time=0.1, sustain_level=0.5, sustain_time=0.5, release_time=0.1):
    """
    Applies an ADSR envelope to an audio signal.

    Args:
    audio_signal (numpy array): Input audio signal.
    attack_time (float): Duration of the attack phase (in seconds).
    decay_time (float): Duration of the decay phase (in seconds).
    sustain_level (float): Amplitude of the sustain phase (between 0 and 1).
    sustain_time (float): Duration of the sustain phase (in seconds).
    release_time (float): Duration of the release phase (in seconds).

    Returns:
    numpy array: Enveloped audio signal.
    """
    n_samples = len(audio_signal)
    attack_samples = int(attack_time * fs)
    decay_samples = int(decay_time * fs)
    sustain_samples = int(sustain_time * fs)
    release_samples = int(release_time * fs)
    envelope = np.zeros(n_samples)

    # Attack phase
    attack_slope = 1.0 / attack_samples
    envelope[:attack_samples] = np.arange(attack_samples) * attack_slope
    # Decay phase
    decay_slope = (1.0 - sustain_level) / decay_samples
    envelope[attack_samples:attack_samples + decay_samples] = 1.0 - np.arange(decay_samples) * decay_slope
    # Sustain phase
    envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain_level
    # Release phase
    release_slope = sustain_level / release_samples
    envelope[attack_samples + decay_samples + sustain_samples:] = sustain_level - np.arange(release_samples) * release_slope

    return audio_signal * envelope


#------------------ Additive Synthesizer ---------------------------------------

def additive_synth(frequencies, amplitudes, duration, sample_rate):
    """
    Generates an additive synthesizer waveform by summing sine waves of
    different frequencies and amplitudes.

    Args:
        frequencies (numpy array): Array of frequencies (in Hz).
        amplitudes (numpy array): Array of amplitudes (between 0 and 1).
        duration (float): Duration of the waveform (in seconds).
        sample_rate (int): Sampling rate of the waveform.

    Returns:
        numpy array: Additive synthesizer waveform.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)

    for freq, amp in zip(frequencies, amplitudes):
        waveform += amp * np.sin(2 * np.pi * freq * t)

    return waveform


#------------------ Subtractive Synthesizer ------------------------------------

def subtractive_synth(frequency, cutoff, resonance, duration, sample_rate):
    """
    Generates a subtractive synthesizer waveform by filtering a sine wave
    with a low-pass filter.

    Args:
        frequency (float): Frequency of the sine wave (in Hz).
        cutoff (float): Cutoff frequency of the low-pass filter (in Hz).
        resonance (float): Resonance of the low-pass filter.
        duration (float): Duration of the waveform (in seconds).
        sample_rate (int): Sampling rate of the waveform.

    Returns:
        numpy array: Subtractive synthesizer waveform.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * frequency * t)

    # Create low-pass filter
    nyquist_freq = sample_rate / 2
    normalized_cutoff = cutoff / nyquist_freq
    b, a = signal.butter(4, normalized_cutoff, btype='lowpass')

    # Apply filter to the sine wave
    filtered_wave = signal.filtfilt(b, a, sine_wave)

    # Apply resonance to the filtered wave
    filtered_wave = signal.lfilter([1], [1, -resonance], filtered_wave)

    return filtered_wave


#------------------ Wavetable Synthesizer --------------------------------------

def wavetable_synth(wavetable, frequency, duration, sample_rate):
    """
    Generates a waveform by playing back a wavetable at a specific frequency.

    Args:
        wavetable (numpy array): Array of waveform samples.
        frequency (float): Frequency of the waveform (in Hz).
        duration (float): Duration of the waveform (in seconds).
        sample_rate (int): Sampling rate of the waveform.

    Returns:
        numpy array: Wavetable synthesizer waveform.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)

    phase_increment = frequency / sample_rate
    phase = 0

    for i in range(len(waveform)):
        index = int(phase * len(wavetable))
        waveform[i] = wavetable[index]
        phase += phase_increment
        if phase > 1:
            phase -= 1

    return waveform


#------------------ FM Synthesizer ---------------------------------------------

def fm_synth(carrier_freq, modulator_freq, modulator_index, duration, sample_rate):
    """
    Generates a FM synthesizer sound wave.

    Args:
        carrier_freq (float): Frequency of the carrier wave.
        modulator_freq (float): Frequency of the modulator wave.
        modulator_index (float): Modulation index.
        duration (float): Duration of the sound wave (in seconds).
        sample_rate (int): Sampling rate of the sound wave.

    Returns:
        numpy array: FM synthesizer sound wave.
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    modulator_wave = np.sin(2 * np.pi * modulator_freq * t)
    carrier_wave = np.sin(2 * np.pi * (carrier_freq + modulator_index * modulator_wave) * t)
    
    return carrier_wave


#------------------ AM Synthesizer ---------------------------------------------

def am_synth(carrier_freq, modulator_freq, modulation_depth, duration, sample_rate):
    """
    Generates an AM synthesizer sound wave.

    Args:
        carrier_freq (float): Frequency of the carrier wave.
        modulator_freq (float): Frequency of the modulator wave.
        modulation_depth (float): Modulation depth.
        duration (float): Duration of the sound wave (in seconds).
        sample_rate (int): Sampling rate of the sound wave.

    Returns:
        numpy array: AM synthesizer sound wave.
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    modulator_wave = (1 + modulation_depth * np.sin(2 * np.pi * modulator_freq * t))
    carrier_wave = np.sin(2 * np.pi * carrier_freq * t) * modulator_wave
    
    return carrier_wave


#------------------ Particle Synthesizer ---------------------------------------

def particle_synth(particle_count, duration, sample_rate):
    """
    Generates a particle synthesizer sound wave.

    Args:
        particle_count (int): Number of particles.
        duration (float): Duration of the sound wave (in seconds).
        sample_rate (int): Sampling rate of the sound wave.

    Returns:
        numpy array: Particle synthesizer sound wave.
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    particle_pos = np.random.rand(particle_count) * duration
    particle_speed = np.random.normal(1, 0.5, particle_count)
    particle_amp = np.random.rand(particle_count) * 2 - 1

    particle_wave = np.zeros_like(t)
    for i in range(particle_count):
        particle_wave += particle_amp[i] * np.sin(2 * np.pi * (t - particle_pos[i]) * particle_speed[i])

    return particle_wave


#------------------ Physical Modeling Synthesis --------------------------------

def karplus_strong(audio_signal, delay_time, decay_rate, fs):
    """
    Applies the Karplus-Strong algorithm to an audio signal for plucked string synthesis.
    The algorithm creates a delay line with the specified delay time and applies a
    decay rate to the signal in the delay line.

    Args:
        audio_signal (numpy array): Input audio signal.
        delay_time (float): Delay time (in seconds) for the delay line.
        decay_rate (float): Decay rate (between 0 and 1) for the delay line.
        fs (int): Sampling rate of the audio signal.

    Returns:
        numpy array: Synthesized audio signal.
    """
    delay_samples = int(delay_time * fs)
    delay_line = np.zeros(delay_samples)
    output_signal = np.zeros(len(audio_signal))

    for i in range(len(audio_signal)):
        output_signal[i] = delay_line[0]
        delay_line = np.append(audio_signal[i] + delay_line[:-1] * decay_rate, 0)

    return output_signal


#------------------ Bird Sound -------------------------------------------------------

def generate_bird_sound(duration=1.0, frequency=1000.0, bandwidth=100.0):
    """
    Generate a simulated bird sound using a chirp signal with a Gaussian amplitude envelope.

    Args:
    duration: float, the duration of the bird call in seconds. Default is 1.0 second.
    frequency: float, the center frequency of the bird call in Hz. Default is 1000.0 Hz.
    bandwidth: float, the bandwidth of the bird call in Hz. Default is 100.0 Hz.

    Returns:
    bird_call: ndarray, the simulated bird call signal with shape (N,), where N is the number of samples.
    """
    # Define the time array for the chirp signal
    t = np.linspace(0, duration, int(duration * 44100), endpoint=False)

    # Generate a chirp signal with a Gaussian amplitude envelope
    chirp_signal = signal.chirp(t, f0=frequency - bandwidth / 2, f1=frequency + bandwidth / 2, t1=duration, method='linear')
    envelope = np.exp(-50 * ((t - duration / 2) / duration) ** 2)
    bird_call = chirp_signal * envelope

    # Normalize the amplitude to the maximum value of 1
    bird_call /= np.max(np.abs(bird_call))

    return bird_call


def detect_bird_sound(audio_signal, threshold=0.5):
    """
    Detect bird sound in a given signal using a peak detection algorithm.

    Args:
    audio_signal: ndarray, the input signal to detect bird calls from.
    threshold: float, the threshold value for peak detection, ranging from 0 to 1. Default is 0.5.

    Returns:
    peaks: ndarray, the sample indices of the detected peaks in the input signal.
    """
    # Find the peaks in the absolute value of the input signal
    peaks, _ = signal.find_peaks(np.abs(audio_signal), height=threshold)

    return peaks


#------------------ Guitar -----------------------------------------------------

def guitar_model(duration=2, sr=44100, f_low=82.4, f_high=1318.5):
    """
    Generate an audio signal of a guitar sound.

    Args:
    duration: float, the duration of the audio signal (in seconds). Default is 2 seconds.
    sr: int, the sample rate of the audio signal. Default is 44100 Hz.
    f_low: float, the lowest frequency of the guitar sound. Default is 82.4 Hz (E2).
    f_high: float, the highest frequency of the guitar sound. Default is 1318.5 Hz (E6).

    Returns:
    audio_signal: ndarray, the audio signal of the guitar sound, with shape (N,), where N is the number of samples.
    """
    # Generate the plucking function
    pluck_func = signal.triang(int(sr/f_low)) * np.random.randn(1, int(sr/f_low))

    # Generate the string function
    string_func = np.zeros((1, int(sr * duration)))
    for f in np.geomspace(f_low, f_high, num=20):
        string_func += np.sin(2 * np.pi * f * np.arange(int(sr * duration)) / sr)

    # Convolve the plucking function and string function
    audio_signal = signal.fftconvolve(pluck_func, string_func, mode='full')[0]

    # Normalize the signal
    audio_signal = audio_signal / np.max(np.abs(audio_signal))

    return audio_signal


#------------------ Piano ------------------------------------------------------

def generate_piano_sound(duration=1.0, sampling_rate=44100, f0=440.0):
    """
    Generate a piano sound using string simulation.

    Args:
    duration: float, the duration of the sound in seconds. Default is 1.0 second.
    sampling_rate: int, the sampling rate of the sound. Default is 44100 Hz.
    f0: float, the fundamental frequency of the string. Default is 440.0 Hz.

    Returns:
    sound: ndarray, the generated piano sound.
    """
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    f = f0 * 2**(np.arange(49)/12)  # Calculate the frequencies of the harmonics
    a = 1 / (1 + 0.01*(f/f0-1)**2)  # Calculate the damping factor for each harmonic
    b = np.zeros_like(f)
    b[0] = 1  # Pluck the string at its center

    # Calculate the impulse response of the string using IIR filtering
    r = 0.99  # Reflection coefficient
    num, den = signal.butter(4, 2*r*np.pi*f[0]/sampling_rate, 'lowpass', analog=True)
    for i in range(1, len(f)):
        # Add a reflection at the bridge
        num_b, den_b = signal.butter(4, 2*r*np.pi*f[i]/sampling_rate, 'lowpass', analog=True)
        num = np.convolve(num, num_b)
        den = np.convolve(den, den_b) + np.polymul([1, -r], np.convolve(num_b, num_b[::-1]))

    # Excite the string with the plucking signal
    x = np.zeros_like(t)
    x[:len(b)] = b

    # Simulate the string vibration using IIR filtering
    y = signal.lfilter(num, den, x)

    # Mix the harmonics and apply damping
    sound = np.sum(a * np.sin(2*np.pi*f.reshape(-1, 1)*t.reshape(1, -1)), axis=0)

    # Normalize the sound to the maximum amplitude of 0.9
    sound *= 0.9 / np.max(np.abs(sound))

    return sound


#------------------ Violin -----------------------------------------------------

def violin_sound(freq, duration, volume):
    """
    Generate a simulated sound of a violin note.

    Args:
    freq: float, the frequency of the note in Hz.
    duration: float, the duration of the note in seconds.
    volume: float, the volume of the note. The range is from 0 to 1.

    Returns:
    signal: ndarray, the simulated sound signal of the violin note.
    """

    # Define the parameters of the violin sound
    sampling_rate = 48000  # Sampling rate in Hz
    attack_time = 0.02     # Attack time in seconds
    decay_time = 0.1       # Decay time in seconds
    sustain_time = 0.5     # Sustain time in seconds
    release_time = 0.3     # Release time in seconds
    harmonics = 20         # Number of harmonics

    # Generate the time vector
    t = np.linspace(0, duration, int(sampling_rate * duration))

    # Generate the amplitude envelope
    envelope = np.zeros_like(t)
    envelope[:int(attack_time * sampling_rate)] = np.linspace(0, 1, int(attack_time * sampling_rate))
    envelope[int(attack_time * sampling_rate):int((attack_time + decay_time) * sampling_rate)] = \
        np.linspace(1, 0.7, int(decay_time * sampling_rate))
    envelope[int((attack_time + decay_time) * sampling_rate):int((attack_time + decay_time + sustain_time) * sampling_rate)] = 0.7
    envelope[int((attack_time + decay_time + sustain_time) * sampling_rate):int((attack_time + decay_time + sustain_time + release_time) * sampling_rate)] = \
        np.linspace(0.7, 0, int(release_time * sampling_rate))

    # Generate the harmonics
    harmonics_amp = np.zeros((len(t), harmonics))
    for h in range(1, harmonics+1):
        harmonics_amp[:, h-1] = np.sin(2 * np.pi * h * freq * t)

    # Apply the amplitude envelope to the harmonics
    harmonics_amp *= envelope[:, np.newaxis]

    # Sum the harmonics and normalize the signal
    signal = np.sum(harmonics_amp, axis=1) / harmonics_amp.shape[1]

    # Apply a low-pass filter to remove high frequency noise
    b, a = signal.butter(5, 0.2, btype='lowpass')
    signal = signal.lfilter(b, a, signal)

    # Scale the signal by the volume
    signal *= volume

    # Normalize the signal
    signal /= np.max(np.abs(signal))

    return signal


#------------------ Cello ------------------------------------------------------

def cello_instrument_model(f0, fs, duration):
    """
    Simulate the sound of a cello using an instrument model.

    Args:
    f0: float, the fundamental frequency of the cello in Hz.
    fs: int, the sample rate in Hz.
    duration: float, the duration of the sound in seconds.

    Returns:
    ndarray, the simulated cello sound.
    """
    # Define the string and body filter parameters
    L = 0.7       # Length of the string in meters
    T = 80        # Tension in the string in N
    rho = 7850    # Density of the string in kg/m^3
    A = 1e-5      # Cross-sectional area of the string in m^2
    E = 2e11      # Young's modulus in N/m^2
    I = A**2 / 12 # Moment of inertia of the string in m^4
    c = np.sqrt(T/(rho*A)) # Wave speed in the string in m/s
    k = 1/fs      # Time step in seconds
    N = int(fs*duration) # Total number of samples

    # Calculate the grid spacing in the string
    h = np.sqrt((c**2 * k**2 + np.sqrt(c**4 * k**4 + 16 * T**2 * k**2 / rho**2)) / 2)

    # Calculate the number of grid points
    J = int(np.ceil(L/h))

    # Calculate the grid spacing again to fit an integer number of grid points
    h = L / J

    # Define the string state vectors
    u = np.zeros(J)
    u_prev = np.zeros(J)

    # Define the body filter coefficients
    f_res = f0 * 2**(1/6) # Resonant frequency of the body in Hz
    Q = 10                # Quality factor of the body filter
    b, a = signal.iirpeak(f_res, Q, fs=fs)

    # Initialize the output array
    y = np.zeros(N)

    # Simulate the sound of the cello
    for n in range(N):
        # Calculate the string state at the next time step
        u_next = 2*u - u_prev + (c*k/h)**2 * (np.roll(u, 1) - 2*u + np.roll(u, -1))
        u_prev = u
        u = u_next

        # Calculate the sound of the body filter
        y[n] = np.sum(b * u) / np.sum(a)

    return y




