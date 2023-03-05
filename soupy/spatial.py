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
from sklearn.decomposition import FastICA
import torch
import torch.nn as nn


#------------------ Stereo sound field enhancement algorithm -------------------

def enhance_stereo(audio_file, delay=0.05, attenuation=0.2):
    """
    Enhance the spatial perception of a stereo audio signal using an acoustic model.

    Args:
    audio_file: str, the path of the stereo audio file to be enhanced.
    delay: float, the delay time (in seconds) of the sound. Default is 0.05 seconds.
    attenuation: float, the attenuation coefficient, ranging from 0 to 1. A larger value means a stronger attenuation. Default is 0.2.

    Returns:
    sr: int, the sample rate of the audio file.
    enhanced_stereo_signal: ndarray, the enhanced stereo audio signal, with shape (N, 2), where N is the number of samples.
    """
    # Load the stereo audio signal
    sr, stereo_signal = wavfile.read(audio_file)
    left = stereo_signal[:, 0]
    right = stereo_signal[:, 1]

    # Calculate the new sound field
    new_left = left + attenuation * np.roll(right, int(delay * sr))
    new_right = right + attenuation * np.roll(left, -int(delay * sr))

    # Combine the new stereo audio signal
    enhanced_stereo_signal = np.stack([new_left, new_right], axis=1)

    return sr, enhanced_stereo_signal


#------------------ Stereo separation algorithm -------------------------------

def separate_stereo_ICA(audio_file):
    """
    Separates the two channels of a stereo audio file using ICA algorithm.

    Args:
    audio_file: str, path to the stereo audio file to be separated.

    Returns:
    sr: int, sampling rate of the audio file.
    left_signal: ndarray, left channel audio signal with shape (N,), where N is the number of samples.
    right_signal: ndarray, right channel audio signal with shape (N,), where N is the number of samples.
    """
    # Load the stereo audio file.
    sr, stereo_signal = wavfile.read(audio_file)

    # Save left and right channel signals as a matrix X, where the first column is the left channel signal and the
    # second column is the right channel signal.
    X = stereo_signal.astype(float)

    # Preprocessing: mean subtraction and variance normalization.
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    # Perform ICA separation.
    ica = FastICA(n_components=2)
    S = ica.fit_transform(X)

    # Separate left and right channel signals.
    left_signal = S[:, 0]
    right_signal = S[:, 1]

    return sr, left_signal, right_signal


def separate_stereo_matrix(audio_file):
    """
    Separate the two channels of a stereo audio using the mixing matrix method.

    Args:
    audio_file: str, path of the stereo audio file to be separated.

    Returns:
    sr: int, sample rate of the audio file.
    left_signal: ndarray, the left channel audio signal with shape (N,),
                 where N is the number of samples.
    right_signal: ndarray, the right channel audio signal with shape (N,),
                  where N is the number of samples.
    """

    # Load the stereo audio
    sr, stereo_signal = wavfile.read(audio_file)

    # Save the left and right channels as a matrix X,
    # where the first column is the left channel signal
    # and the second column is the right channel signal
    X = stereo_signal.astype(float)

    # Preprocessing: mean subtraction and variance normalization
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    # Calculate the covariance matrix
    cov = np.cov(X.T)

    # Calculate the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Calculate the mixing matrix A
    A = np.vstack((eigenvectors[:, 1], eigenvectors[:, 0]))

    # Separate the left and right channel signals
    S = np.dot(A, X.T)
    left_signal = S[0, :]
    right_signal = S[1, :]

    return sr, left_signal, right_signal


class SeparatorNeuralNetwork(nn.Module):
    def __init__(self):
        super(SeparatorNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def separate_stereo_neural_network(audio_file, model_file):
    """
    Separate the two channels of a stereo audio using the neural network method.

    Args:
    audio_file: str, path of the stereo audio file to be separated.
    model_file: str, path of the trained PyTorch model file.

    Returns:
    sr: int, sample rate of the audio file.
    left_signal: ndarray, the left channel audio signal with shape (N,),
                 where N is the number of samples.
    right_signal: ndarray, the right channel audio signal with shape (N,),
                  where N is the number of samples.
    """

    # Load the stereo audio
    sr, stereo_signal = wavfile.read(audio_file)

    # Save the left and right channels as a matrix X,
    # where the first column is the left channel signal
    # and the second column is the right channel signal
    X = stereo_signal.astype(float)

    # Preprocessing: mean subtraction and variance normalization
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)

    # Load the trained PyTorch model
    model = SeparatorNeuralNetwork()
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    # Separate the left and right channel signals using the model
    X = torch.tensor(X)
    S = model(X).detach().numpy()
    left_signal = S[:, 0]
    right_signal = S[:, 1]

    return sr, left_signal, right_signal


#------------------ Multichannel mixing algorithm -------------------------------

def multichannel_mixing(signals, weights):
    """
    Mix multiple audio signals together.

    Parameters:
    signals: numpy array with shape (N,M), representing M audio signals, each with N samples.
    weights: numpy array with shape (M,), representing the weight of each audio signal.

    Returns:
    mixed_signal: numpy array with shape (N,), representing the mixed audio signal.
    """
    mixed_signal = np.sum(signals * weights.reshape(-1, 1), axis=0)
    return mixed_signal


#------------------ Spatial audio encoding algorithm -------------------------------

def spatial_audio_encoding(audio_signals, mic_array_positions, source_position):
    """
    Encode multichannel audio signals into a spatial audio format.

    Args:
    audio_signals: Multichannel audio signals, each channel's data is a 1D numpy array.
    mic_array_positions: The positions of each microphone in the array, a numpy array of shape (N, 3), where N is the number of microphones and each row represents the position coordinates of a microphone.
    source_position: The position coordinates of the source signal in 3D space, a numpy array of length 3.

    Returns:
    encoded_signals: The encoded audio signals, a 1D numpy array.
    """

    # Calculate the distances from each microphone to the source signal.
    distances = np.sqrt(np.sum((mic_array_positions - source_position)**2, axis=1))

    # Calculate the weights of each microphone based on their distances.
    weights = 1 / distances

    # Sum the weighted signals for each channel to obtain the encoded signal.
    encoded_signal = np.sum(weights * audio_signals, axis=0)

    return encoded_signal


#------------------ Spatial audio restoration algorithm -------------------------------

def spatial_audio_decoding(encoded_audio, transform_matrix):
    '''
    Decode spatial audio signals

    Args:
    encoded_audio (ndarray): Encoded spatial audio signal with shape (n_frames, n_channels)
    transform_matrix (ndarray): Transformation matrix used during spatial audio encoding with shape (n_channels, n_spatial_features)

    Returns:
    decoded_audio (ndarray): Decoded multi-channel audio signal with shape (n_frames, n_channels)
    '''

    decoded_audio = np.matmul(encoded_audio, np.transpose(transform_matrix))
    return decoded_audio





