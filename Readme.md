# 🎭 Yumeka - The Dream Voice

## 🌟 Overview

Yumeka (from Japanese 夢 "yume" meaning "dream" and 歌 "ka" meaning "song/voice") is a cutting-edge AI-powered voice generation system that seamlessly combines natural language understanding with high-quality text-to-speech synthesis. It transforms any text input into rich, expressive audio that feels natural and conversational.


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- PyTorch
- CUDA-capable GPU (optional, but recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/daviduche03/yumeka.git
cd yumeka
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. run:
```bash
cd yumeka
python yumeka.py
```

### Basic Usage

```python
from yumeka import Yumeka
import torch

# Initialize Yumeka
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Yumeka(device)

# Generate audio from text
audio_arr, sample_rate, output_text = model.generate(
    "Tell me about artificial intelligence."
)

# Save the generated audio
model.save_audio(audio_arr, sample_rate, "output.wav")
```



## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for their incredible transformers library
- [ParlerTTS](https://huggingface.co/parler-tts) for the TTS foundation
- The open-source community for continuous inspiration and support

## 📬 Contact

Project Link: [https://github.com/daviduche03/yumeka](https://github.com/daviduche03/yumeka)

---
Made with ❤️ by the Me