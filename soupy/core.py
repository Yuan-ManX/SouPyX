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
from scipy.io import wavfile
from scipy.io import flac
from scipy.io import ogg
from scipy import signal
from scipy.fftpack import dct
from scipy.io.wavfile import write
import mido
import os
import subprocess


#------------------ Utility Functions ------------------------------------------

def read(audio_file, sr=None):
    """
    Read mono audio file.

    Args:
    audio_file (str): Path of audio file to read.
    sr (int): Sampling rate. If None, the file's own sampling rate will be used.

    Returns:
    sr (int): Sampling rate.
    audio_signal (ndarray): Read audio signal with shape (N,), where N is number of samples.
    """
    # Get the file extension
    file_ext = audio_file.split('.')[-1]

    # Call different read functions depending on the file extension
    if file_ext == 'wav':
        sr, audio_signal = wavfile.read(audio_file)
    elif file_ext == 'flac':
        audio_signal, sr = flac.read(audio_file)
    elif file_ext == 'ogg':
        audio_signal, sr = ogg.read(audio_file)
    else:
        raise ValueError(f"Unsupported audio file format: {file_ext}")

    # If the sampling rate is specified, resample the audio signal
    if sr is not None and sr != audio_signal.shape[0]:
        audio_signal = signal.resample(audio_signal, sr, audio_signal.shape[0])

    # If the audio is stereo, convert to mono
    if len(audio_signal.shape) > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    return sr, audio_signal


def write(audio_signal, audio_file, sr=44100):
    """
    Write mono audio signal to audio file.

    Args:
    audio_signal (ndarray): Audio signal to write with shape (N,), where N is number of samples.
    audio_file (str): Path of audio file to write.
    sr (int): Sampling rate. Default is 44100.

    Returns:
    None
    """
    # Get the file extension
    file_ext = audio_file.split('.')[-1]

    # Call different write functions depending on the file extension
    if file_ext == 'wav':
        write(audio_file, sr, audio_signal.astype(np.int16))
    elif file_ext == 'flac':
        flac.write(audio_file, sr, audio_signal.astype(np.int16))
    elif file_ext == 'ogg':
        ogg.write(audio_file, sr, audio_signal.astype(np.int16))
    else:
        raise ValueError(f"Unsupported audio file format: {file_ext}")


#------------------ MIDI -------------------------------------------------------

def midi_to_audio(midi_data, fs=44100, duration=5):
    """
    Convert MIDI data to audio signal and save as a .wav file.

    Args:
    midi_data (numpy.ndarray): 2D array containing the MIDI data, with each row representing a note, including start time, end time, pitch, velocity, etc.
    fs (int): Audio sampling rate, default is 44100Hz.
    duration (float): Duration of the audio, default is 5 seconds.

    Returns:
    None
    """
    # Set synthesizer parameters
    n_samples = int(fs * duration)
    t = np.linspace(0, duration, n_samples, False)
    audio = np.zeros(n_samples)

    # Synthesize audio signal
    for note in midi_data:
        start, end, pitch, velocity = note
        frequency = 2 ** ((pitch - 69) / 12) * 440
        waveform = velocity * np.sin(2 * np.pi * frequency * t)
        start_sample = int(start * fs)
        end_sample = min(int(end * fs), n_samples)
        audio[start_sample:end_sample] += waveform[:end_sample - start_sample]

    # Normalize audio signal
    audio /= np.max(np.abs(audio))

    # Save audio as .wav file
    write('midi_to_audio.wav', fs, audio)


def audio_to_midi(audio_data, fs=44100):
    """
    Convert audio data to MIDI data and save as MIDI file.

    Args:
    audio_data (numpy.ndarray): One-dimensional or two-dimensional array containing audio data. One dimension represents mono audio, and two dimensions represent stereo audio. The value range is [-1, 1].
    fs (int): Audio sample rate, default is 44100Hz.

    Returns:
    None
    """

    # Convert stereo audio to mono if necessary
    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize audio signal
    audio_data /= np.max(np.abs(audio_data))

    # Set threshold for note detection
    threshold = 0.2

    # Detect note onsets and offsets
    note_onsets = []
    note_offsets = []
    for i in range(1, len(audio_data)):
        if audio_data[i] > threshold and audio_data[i-1] <= threshold:
            note_onsets.append(i)
        elif audio_data[i] <= threshold and audio_data[i-1] > threshold:
            note_offsets.append(i)

    # Create MIDI file
    midi_file = mido.MidiFile(type=1)

    # Create MIDI track
    track = mido.MidiTrack()

    # Add note messages to MIDI track
    for i in range(len(note_onsets)):
        note_start = note_onsets[i] / fs
        note_end = note_offsets[i] / fs
        note_duration = note_end - note_start
        note_pitch = int(np.random.uniform(60, 80))
        note_velocity = int(np.random.uniform(80, 100))
        note_on_message = mido.Message('note_on', note=note_pitch, velocity=note_velocity, time=int(note_start*1000))
        note_off_message = mido.Message('note_off', note=note_pitch, velocity=note_velocity, time=int(note_duration*1000))
        track.append(note_on_message)
        track.append(note_off_message)

    # Add track to MIDI file
    midi_file.tracks.append(track)

    # Save MIDI file
    midi_file.save('audio_to_midi.mid')


#------------------ Audio Conversions ------------------------------------------

def audio_format_conversion(input_file, output_format):
    """
    Convert an audio file from one format to another.

    Args:
    input_file (str): The path of the input audio file, can be either an absolute or relative path.
    output_format (str): The desired output audio file format, e.g. 'wav', 'mp3', 'ogg', etc.

    Returns:
    None
    """
    # Get basic information of the input file
    input_path, input_extension = os.path.splitext(input_file)
    sample_rate, audio_data = wavfile.read(input_file)

    # Normalize the audio data to the range [-1, 1]
    audio_data = audio_data.astype(np.float32)
    audio_data /= np.max(np.abs(audio_data))

    # Create the output file name
    output_file = input_path + '.' + output_format

    # Define the FFmpeg command-line arguments
    cmd = ['ffmpeg', '-y', '-f', 's16le', '-ar', str(sample_rate), '-ac', '1', '-i', '-', '-acodec', 'lib' + output_format, output_file]

    # Call FFmpeg to perform audio format conversion
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        process.stdin.write(audio_data.tobytes())
        process.stdin.close()
        process.communicate()

    # Print a message to indicate the completion of the conversion
    print('Conversion from', input_extension[1:], 'to', output_format, 'completed successfully.')


def freq_to_mel(freq):
    """
    Convert frequency to Mel frequency.

    Args:
    freq: frequency

    Returns:
    Corresponding Mel frequency
    """
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def mel_to_freq(mel_freq):
    """
    Convert Mel frequency to frequency.

    Args:
    mel_freq: Mel frequency

    Returns:
    Corresponding frequency
    """
    return 700.0 * (10.0**(mel_freq / 2595.0) - 1.0)


def get_mel_filterbanks(num_filters, nfft, samplerate):
    """
    Calculate Mel filter banks.

    Args:
    num_filters: Number of Mel filter banks
    nfft: FFT length
    samplerate: Sampling rate

    Returns:
    Mel filter banks
    """
    # Calculate the frequency range of the FFT
    low_freq_mel = 0
    high_freq_mel = freq_to_mel(samplerate / 2)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)

    # Convert Mel frequencies to linear frequencies and perform FFT
    hz_points = mel_to_freq(mel_points)
    bin_points = np.floor((nfft + 1) * hz_points / samplerate)

    # Create Mel filter banks
    fbank = np.zeros((num_filters, int(nfft / 2 + 1)))
    for m in range(1, num_filters + 1):
        f_m_minus = int(bin_points[m - 1])
        f_m = int(bin_points[m])
        f_m_plus = int(bin_points[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

    return fbank


#------------------ Time Conversions -------------------------------------------

def time_pitch_shift(audio_data, sr, pitch_shift_factor):
    """
    Pitch-shifts an audio signal while maintaining the duration.

    Args:
    audio_data (numpy.ndarray): One-dimensional or two-dimensional array containing audio data. 
                                One dimension represents mono audio, and two dimensions represent stereo audio. 
                                The value range is [-1, 1].
    sr (int): Audio sampling rate in Hz.
    pitch_shift_factor (float): Shift factor. A value of 1 represents no pitch shift, 
                                values less than 1 represent pitch lowering, 
                                and values greater than 1 represent pitch raising.

    Returns:
    numpy.ndarray: Pitch-shifted audio data.
    """

    # Convert stereo audio to mono if necessary
    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Normalize audio signal
    audio_data /= np.max(np.abs(audio_data))

    # Define constants
    window_size = 2048
    hop_size = 512
    n_fft = window_size
    win = np.hanning(window_size)
    pad_length = n_fft - hop_size
    freq_scale = np.linspace(0, sr, n_fft)
    phase_advance = np.linspace(0, np.pi * hop_size, window_size//2 + 1)

    # Compute STFT of audio data
    stft = np.array([np.fft.rfft(win * audio_data[i:i+window_size], n_fft) for i in range(0, len(audio_data)-window_size, hop_size)])
    mag = np.abs(stft)
    phase = np.angle(stft)

    # Perform frequency-domain phase-shift
    bin_offset = np.round((freq_scale * pitch_shift_factor - freq_scale)).astype(int)
    stft_shifted = np.zeros(stft.shape, dtype=np.complex)
    for i in range(0, stft.shape[0]):
        for j in range(0, stft.shape[1]):
            j2 = j + bin_offset[j]
            if j2 >= 0 and j2 < stft.shape[1]:
                stft_shifted[i,j2] = mag[i,j] * np.exp(1j * phase[i,j] - 1j * phase_advance[j2])

    # Perform inverse STFT to obtain time-domain signal
    y_shifted = np.zeros((stft_shifted.shape[0]-1) * hop_size + window_size + pad_length)
    for i in range(stft_shifted.shape[0]):
        ytmp = np.fft.irfft(stft_shifted[i])
        y_shifted[i*hop_size:i*hop_size+window_size] += win * ytmp
    y_shifted = np.delete(y_shifted, range(pad_length))

    # Normalize audio signal
    y_shifted /= np.max(np.abs(y_shifted))

    return y_shifted






