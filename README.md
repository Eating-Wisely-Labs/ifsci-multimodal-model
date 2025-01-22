# IFS Multimodal Model

The IFS Multimodal Model is designed to be a versatile tool for Food Science vision-language tasks, including image understanding, visual reasoning, and conversation. It's specifically designed for the IFS team and community. The model will be release soon.  A powerful multimodal model combining vision and language capabilities, built with LLaVA and LoRA techniques. To combat the OpenAI's limitation, we are building this model for better product precision (calorie, nutrition, etc.) and better customer experience.


## 🌟 Features

- Vision-Language foundation model based on LLaVA architecture
- Efficient fine-tuning using LoRA (Low-Rank Adaptation)
- Multi-task capabilities: image understanding, visual reasoning, and conversation
- Optimized performance with reduced parameter count
- Support for various vision-language tasks

## 🛠️ Requirements

torch>=2.0.0
transformers==4.45.2
accelerate>=0.20.0
peft>=0.3.0
bitsandbytes>=0.39.0
sentencepiece>=0.1.97

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/ifs_multimodal_model.git
cd ifs_multimodal_model
pip install -r requirements.txt
```

### Customize Your Own VQA Dataset for Model Training

```python
# Initialize the LlavaLoraFt instance
llava_ft = LlavaLora(
    model_name="llava-hf/llava-v1.6-mistral-7b-hf",
    output_dir="./llava-finetuned"
)

# Prepare your training data
train_images = ["llava_v1_5_radar.jpg"]
train_prompts = ["What is shown in this image?"]
train_responses = ["This is a radar image showing..."]

# Initialzie IFS data holder, where the data is annotated by IFS users.
# ifs_data_holder = IFSAnnotationData(db_path="./ifs_annotation.db", image_base_path="./ifs_images")
# train_images, train_prompts, train_responses = ifs_data_holder.prepare_dataset()

# Create dataset
train_dataset = llava_ft.prepare_dataset(train_images, train_prompts, train_responses)

# Start training
llava_ft.train(
    train_dataset=train_dataset,
    num_epochs=3,
    batch_size=1,
    learning_rate=2e-4
)

# Save the model
save_directory = "./llava-finetuned-final"
llava_ft.save_model(save_directory)

# Perform inference with the trained model
test_image_path = "ifs_banner.jpeg"
test_prompt = "What is shown in this image?"
)
```

### Inference

```python
from ifs_llava_lora import LlavaLora

# Load the saved model for inference
inference_model = LlavaLora.load_model(save_directory)

# Generate response
response = inference_model.generate_response(test_image_path, test_prompt)
print(f"Model response: {response}")
```

## 📊 Performance（Ongoing）

| Task | Metric | Score |
|------|--------|-------|


## 🔧 Advanced Configuration

The model supports various configuration options:

- Adjustable LoRA parameters
- Mixed precision training
- Gradient accumulation
- Custom training strategies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- IFS team for the LlavaLoRA Fine-tuning implementation
- IFS team and community for the annotation data

## 📮 Contact

For questions and feedback:
- Create an issue in this repository
- X: @0xSakataGintoki @IFS_AI
