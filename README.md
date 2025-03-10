# 🖼️📝 MultiModal Vision-Language Model: SigLIP + PaliGemma

This project implements a powerful multimodal model that combines **SigLIP Vision Transformer** and **PaliGemma Language Model** for tasks like image captioning, visual question answering, and conditional text generation. The model seamlessly integrates vision and language processing, enabling it to understand and generate text based on visual inputs.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30%2B-yellow)](https://huggingface.co/transformers/)

---

## 🌟 Features

- **Multimodal Integration**: Combines **SigLIP Vision Transformer** and **PaliGemma Language Model** for tasks involving both images and text.
- **Flexible Architecture**: Supports custom configurations for vision and language components.
- **Easy-to-Use**: Provides simple APIs for loading the model, preprocessing inputs, and running inference.
- **Efficient Inference**: Optimized for both CPU and GPU usage.

---

## 🚀 Quick Start

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AydinXudiyev/MultiModal_Vision_Transormation_Model-Siglip-PaligemmaModel-.git
   cd MultiModal_Vision_Transormation_Model-Siglip-PaligemmaModel-
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model weights and configuration file into the `model/` directory.

### Usage

Here's how to load the model and generate text based on an image and a prompt:

```python
from model_loader import load_hf_model
from PIL import Image

# Load the model and tokenizer
model_path = "model/"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_hf_model(model_path, device)

# Load and preprocess the image
image = Image.open("path/to/image.jpg")

# Tokenize the text prompt
prompt = "What is in this image?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate text
with torch.no_grad():
    outputs = model.generate(pixel_values=image, input_ids=inputs["input_ids"])
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

---

## 🧠 Model Architecture

The model consists of two main components:

1. **SigLIP Vision Transformer (ViT):**
   - Processes image inputs using a patch-based approach.
   - Extracts visual features with a transformer encoder.

2. **PaliGemma Language Model:**
   - Processes text inputs.
   - Generates text based on combined visual and textual features.

The two components are integrated through a **multimodal projector**, which aligns visual and textual embeddings.

---

## ⚙️ Configuration

The model's behavior can be customized using the `config.json` file. Key parameters include:

- `hidden_size`: Dimensionality of the model's hidden states.
- `num_hidden_layers`: Number of transformer layers.
- `num_attention_heads`: Number of attention heads.
- `image_size`: Size of the input image (height and width).
- `patch_size`: Size of the patches used in the vision transformer.

---

## 📂 Repository Structure

```
MultiModal_Vision_Transormation_Model-Siglip-PaligemmaModel-/
├── model/                   # Directory for model weights and config
│   ├── config.json          # Model configuration
│   ├── model.safetensors    # Model weights
├── model_loader.py          # Script to load the model
├── modelling_gemma.py       # Implementation of the Gemma model
├── modelling_siglip.py      # Implementation of the SigLIP model
├── requirements.txt         # List of dependencies
├── README.md                # This file
└── LICENSE                  # License file
```

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

For major changes, please open an issue first to discuss your ideas.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thanks to [Hugging Face](https://huggingface.co/) for the `transformers` library.
- Inspired by the original [SigLIP](https://arxiv.org/abs/xxxx.xxxx) and [Gemma](https://arxiv.org/abs/xxxx.xxxx) papers.

---

## 📧 Contact

If you have any questions or suggestions, feel free to reach out:

- **GitHub**: [AydinXudiyev](https://github.com/AydinXudiyev)
- **Email**: [aydinxudiyev75@gmail.com](mailto:aydinxudiyev75@gmail.com)
