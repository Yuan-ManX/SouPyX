<p align="center">
  <img src="SouPyX.png" alt="SouPyX" style="display:block; margin:auto; transform: scale(1.0);" />
</p>

## SouPyX (Sound Python Explore): An Audio Exploration Space.🪐

SouPyX is a very colourful space for audio exploration, suitable for research and exploration in a variety of audio fields. In SouPyX you can carry out research and exploration in audio processing, sound synthesis, audio effects, spatial audio, audio visualisation, AI audio and much more.

- [中文版](./README_CN.md)
- [English](./README.md)

SouPyX provides a number of useful audio processing tools, including audio file reading and output, format conversion, MIDI conversion and more, to help developers work with audio data in a more flexible way. In addition, SouPyX provides a range of sound synthesis methods such as oscillators, ADSR envelopes, additive synthesis, subtractive synthesis, wavetable synthesis, particle synthesis, physical modelling, etc., and also provides implementations of some common audio processing algorithms such as filters, equalisers, compressors, reverberation delays, etc. SouPyX also provides a wealth of audio visualisation tools that can convert audio into visual images or animations, allowing developers to more intuitively understand the characteristics and structure of audio.

SouPyX can also be used for research in areas such as AI audio music, by using neural network models for exploring applications in speech, music and sound effects. In addition, SouPyX can be used for research in spatial audio and game development, such as simulating sound propagation and reflection in different environments, enabling spatial audio mixing, localisation, virtual environment creation, etc., bringing more flexible audio exploration tools to developers.

In summary, SouPyX is a comprehensive and rich audio exploration platform for research and applications in a variety of audio domains. Whether you are an audio programmer, music producer, sound designer, AI audio researcher, game developer or audio enthusiast, you can find the right tools and resources in SouPyX to carry out your own research and creation of interest and start your own audio exploration journey.

## Installation

To install the project, use the following command:

```python
pip install SouPyX
```

## Quick Start

First, follow the steps in the Installation section to install the SouPyX package and its dependencies.

* Audio Processing

```python
import SouPyX as spx

# audio read
audio_file_path = 'audio_file.wav'
sr, audio_data = spx.core.read(audio_file_path, sr=44100)

# audio to midi
midi = spx.core.audio_to_midi(audio_data)

# audio conversion
input_file = 'input.wav'
output_format = 'mp3'
spx.core.audio_format_conversion(input_file, output_format)
```

* Oscillator、Filter、Display

```python
import SouPyX as spx

# Generate a triangle wave with a frequency of 440Hz for 1 second
waveform = spx.synths.oscillator(freq=440, duration=1, type='triangle')

# filter
cutoff_freq = 2000
fs=44100
filter_type='lowpass'
filtered_audio = spx.effects.filter(audio_data=waveform, fs=fs, filter_type=filter_type, cutoff_freq=cutoff_freq)

# display
spx.display.waveform(waveform)
spx.display.waveform(filtered_audio)

print(waveform)
print(filtered_audio)
```

## Feature List

* Feature 1: [Core](./SouPyX/core.py) Audio processing functions, including audio file reading and output, audio format conversion, MIDI conversion, audio feature extraction, etc.
* Feature 2: [Synths](./SouPyX/synths.py) Sound synthesis functions, including basic waveforms, oscillators, ADSR, additive synthesis, subtractive synthesis, wavetable synthesis, FM synthesis, AM synthesis, particle synthesis, physical modelling, musical instruments, etc.
* Feature 3: [Effects](./SouPyX/effects.py) Audio effects functions, including filters, compressors, reverbs, delays, modulators, Doppler effects, flanger, chorus, modulation, etc.
* Feature 4: [Spatial Audio](./SouPyX/spatial.py) Spatial audio functions, including stereo sound field enhancement algorithm, stereo separation algorithm, multichannel mixing algorithm, spatial audio encoding algorithm, spatial audio reduction algorithm, etc.
* Feature 5: [Display](./SouPyX/display.py) Audio visualization function, including waveform graph, spectrum graph, sound spectrum graph, waterfall graph, 3D spectrum graph, etc.
* Feature 6: [Models](./SouPyX/models.py) Audio models function, including Markov models, Hidden Markov Models (HMM), Recurrent Neural Networks (RNN), Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), etc.
* Feature 7: More new features are in development！

## Contribution Guide

Everyone is welcome to contribute to this project! If you want to contribute code, follow these steps:

1. Fork this project on GitHub.
2. Create a branch.
3. Commit your changes.
4. Create a pull request.

If you find errors or have any suggestions, submit an issue or propose an improvement on GitHub.

## License ([MIT License](./LICENSE))

This project is licensed under the MIT license.

Copyright (c) 2023 Yuan-Man

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
