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
from scipy.sparse import lil_matrix
from scipy.stats import rv_discrete
import torch
import torch.nn as nn


#------------------------------- Markov ----------------------------------------

def markov_chain(notes, order):
    """
    Music generation algorithm based on Markov model.
    
    Args:
    notes: list of notes
    order: order of Markov chain
    
    Returns:
    generated_notes: generated list of notes
    """
    # Convert note sequence to integer sequence for matrix calculation
    note_to_int = dict((note, i) for i, note in enumerate(set(notes)))
    int_to_note = dict((i, note) for i, note in enumerate(set(notes)))
    notes_int = [note_to_int[note] for note in notes]

    # Create transition matrix for Markov chain
    num_notes = len(set(notes))
    transition_matrix = lil_matrix((num_notes ** order, num_notes))
    for i in range(len(notes_int) - order):
        current_state = tuple(notes_int[i:i+order])
        next_state = notes_int[i+order]
        transition_matrix[current_state, next_state] += 1

    # Normalize transition matrix and compute probability matrix
    row_sums = np.array(transition_matrix.sum(axis=1))[:, 0]
    probability_matrix = transition_matrix / row_sums[:, np.newaxis]

    # Generate new sequence of notes
    current_state = tuple(notes_int[:order])
    generated_notes = list(current_state)
    for i in range(50):
        next_state = np.random.choice(range(num_notes), p=probability_matrix[current_state])
        generated_notes.append(next_state)
        current_state = tuple(generated_notes[-order:])

    # Convert generated note sequence to list of note strings
    generated_notes = [int_to_note[note] for note in generated_notes]

    return generated_notes


#------------------------------- HMM -------------------------------------------

def hmm(notes, n_states, n_samples):
    """
    Generate music using Hidden Markov Model.
    
    Args:
    notes: list of notes
    n_states: number of hidden states in HMM
    n_samples: number of output samples to generate

    Returns:
    generated_notes: generated list of notes
    """
    # convert note sequence to integer sequence
    note_to_int = dict((note, i) for i, note in enumerate(set(notes)))
    int_to_note = dict((i, note) for i, note in enumerate(set(notes)))
    notes_int = np.array([note_to_int[note] for note in notes])

    # create transition probability matrix for hidden states
    A = np.random.rand(n_states, n_states)
    A /= A.sum(axis=1, keepdims=True)

    # create emission probability matrix for visible states
    B = np.random.rand(n_states, len(note_to_int))
    B /= B.sum(axis=1, keepdims=True)

    # create initial state distribution
    pi = np.random.rand(n_states)
    pi /= pi.sum()

    # generate output using HMM
    model = rv_discrete(values=(range(len(note_to_int)), B.shape[0]), name='note')
    states = np.zeros(n_samples, dtype=int)
    observations = np.zeros(n_samples, dtype=int)
    states[0] = np.argmax(np.random.multinomial(1, pi))
    observations[0] = model.rvs(size=1, loc=states[0])[0]
    for t in range(1, n_samples):
        states[t] = np.argmax(np.random.multinomial(1, A[states[t-1], :]))
        observations[t] = model.rvs(size=1, loc=states[t])[0]

    # convert generated integer sequence back to note sequence
    generated_notes = [int_to_note[note] for note in observations]

    return generated_notes


#------------------------------- RNN -------------------------------------------

class RNN(nn.Module):
    """
    Recurrent Neural Network model.

    Args:
    input_size: number of input features
    hidden_size: number of neurons in the hidden layer
    output_size: number of output features
    num_layers: number of layers in the RNN
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, sequence_length, input_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        batch_size = x.size(0)
        rnn_out, hidden = self.rnn(x, hidden)
        rnn_out = rnn_out.reshape(-1, self.hidden_size)
        output = self.fc(rnn_out)
        output = output.view(batch_size, -1, self.output_size)
        return output, hidden

    def init_hidden(self, batch_size):
        # initialize the initial state of the hidden layer
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden


#------------------------------- GAN -------------------------------------------

class GAN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, latent_size, num_layers=1):
        """
        GAN Neural Network model.

        Args:
        input_size: the number of features in the input data
        hidden_size: the number of neurons in the hidden layer
        output_size: the number of features in the output data
        latent_size: the dimension of the latent space
        num_layers: the number of layers in the neural network
        """
        super(GAN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        # Generator Network
        self.generator = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

        # Discriminator network
        self.discriminator = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def generate(self, z):
        """
        Generators forward propagation.

        Args:
        z: noise vector sampled from potential space

        Returns:
        The generated music sequence
        """
        return self.generator(z)

    def discriminate(self, x):
        """
        Discriminator forward propagation.

        Args:
        x: music sequence

        Returns:
        The output of the discriminator
        """
        return self.discriminator(x)

    def forward(self, z):
        """
        Forward propagation

        Args:
        z: the noise vector sampled from the potential space

        Returns:
        The generated music sequence and the output of the discriminator
        """
        x = self.generate(z)
        y = self.discriminate(x)
        return x, y


#------------------------------- VAE -------------------------------------------

class VAE(nn.Module):
    """
    Variational Autoencoder model

    Args:
    input_size: number of features in input data
    hidden_size: number of neurons in hidden layer
    latent_size: dimensionality of latent space
    """
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2)
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    # Reparameterize using mean and standard deviation
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Encoder forward pass
    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var

    # Decoder forward pass
    def decode(self, z):
        x = self.decoder(z)
        return x

    # Forward pass
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var


#------------------------------- Diffusion -------------------------------------

