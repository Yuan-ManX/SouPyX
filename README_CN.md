<p align="center">
  <img src="SouPyX.png" alt="SouPyX" style="display:block; margin:auto;" />
</p>

# SouPyX (Sound Python Explore): 音频探索空间🪐

SouPyX是一个非常丰富多彩的音频探索空间，适用于各种音频领域的研究和探索。在SouPyX中，您可以进行音频处理、声音合成、音频效果、空间音频、音频可视化、AI音频等方面的研究和探索。

SouPyX提供了许多实用的音频处理工具，包括音频文件的读取和输出、格式转换、MIDI转换等，可以帮助开发人员更加灵活地处理音频数据。此外，SouPyX还提供了一系列声音合成方法，如振荡器、ADSR包络、加法合成、减法合成、波表合成、粒子合成、物理建模等，并且还提供了一些常用的音频处理算法的实现，如滤波、均衡器、压缩器、混响延时等。SouPyX还提供了丰富的音频可视化工具，可以将音频转换为可视化图像或动画，让开发人员更加直观地了解音频的特性和结构。

SouPyX还可用于AI音频音乐等方面的研究，通过使用神经网络模型进行语音、音乐和声音效果等方面的应用探索。此外，SouPyX还可用于空间音频和游戏开发等方面的研究，例如模拟不同环境下的声音传播和反射，实现空间音频的混合、定位、虚拟环境建立等，为开发人员带来更加灵活的音频探索工具。

综上所述，SouPyX是一个全面、丰富的音频探索平台，适用于各种音频领域的研究和应用。无论您是音频程序员、音乐制作人、声音设计师、AI音频研究人员、游戏开发者还是音频爱好者，都可以在SouPyX中找到适合自己的工具和资源，开展自己感兴趣的研究和创作，开启自己的音频探索之旅。

## 安装

要安装项目，请使用以下命令：

```python
pip install SouPyX
```

## 快速开始

首先，按照安装部分的步骤来安装SouPyX包和它的依赖项。

* 音频处理

```python
import SouPyX as spx

# 音频读取
audio_file_path = 'audio_file.wav'
sr, audio_data = spx.core.read(audio_file_path, sr=44100)

# 音频文件转为MIDI文件
midi = spx.core.audio_to_midi(audio_data)

# 音频格式转换
input_file = 'input.wav'
output_format = 'mp3'
spx.core.audio_format_conversion(input_file, output_format)

```

* 振荡器、滤波器、波形图

```python
import SouPyX as spx

# 振荡器
waveform = spx.synths.oscillator(freq=440, duration=1, type='triangle')

# 滤波器
cutoff_freq = 2000
fs=44100
filter_type='lowpass'
filtered_audio = spx.effects.filter(audio_data=waveform, fs=fs, filter_type=filter_type, cutoff_freq=cutoff_freq)

# 波形图
spx.display.waveform(waveform)
spx.display.waveform(filtered_audio)

print(waveform)
print(filtered_audio)

```

* SOFA格式

```python
import SouPyX as spx

# 实例化SOFA类并加载SOFA文件
sofa = spx.spatial.SOFA('HRTF.sofa')

# 获取采样频率
sampling_rate = sofa.get_sampling_rate()

# 获取第1个声源的脉冲响应数据
ir = sofa.get_ir(1)

# 获取第1个声源的位置
source_pos = sofa.get_source_position(1)

# 获取监听者的方向向量
listener_orientation = sofa.get_listener_orientation()

# 关闭SOFA文件
sofa.close()

```

## 功能列表

* 功能一：[Core](./SouPyX/core.py)  音频处理部分，包括了音频文件的读取输出、音频格式转换、MIDI转换、音频特征提取等。
* 功能二：[Synths](./SouPyX/synths.py)  声音合成部分，包括了基础波形、振荡器、ADSR、加法合成、减法合成、波表合成、FM合成、AM合成、粒子合成、物理建模、乐器等。
* 功能三：[Effects](./SouPyX/effects.py)  音频效果部分，包括了滤波器、压缩器、混响器、延时器、变调器、多普勒效果器、镶边、合唱、调制等。
* 功能四：[Spatial Audio](./SouPyX/spatial.py)  空间音频部分，包括了立体声声场增强算法、立体声分离算法、多声道混音算法、空间音频编码算法、空间音频还原算法、SOFA音频格式处理等。
* 功能五：[Display](./SouPyX/display.py)  音频可视化部分，包括了波形图、频谱图、声谱图、瀑布图、三维频谱图等。
* 功能六：[Models](./SouPyX/models.py)  音频模型部分，包括了Markov模型、隐马尔可夫模型（HMM）、循环神经网络（RNN）、变分自动编码器（VAE）、生成对抗网络（GAN）等。
* 功能七：更多新功能正在开发中！

## 贡献指南

欢迎任何人为本项目做出贡献！如果您想贡献代码，请遵循以下步骤：

1. 在 GitHub 上 fork 本项目。
2. 创建一个分支。
3. 提交您的更改。
4. 创建一个 pull 请求。

如果您发现了错误或有任何建议，请在 GitHub 上提交问题或提出建议。

## 许可证 ([MIT License](./LICENSE))

本项目采用 MIT 许可证。

Copyright (c) 2023 Yuan-Man

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
