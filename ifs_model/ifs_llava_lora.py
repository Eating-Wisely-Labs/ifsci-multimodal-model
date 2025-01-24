from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, Trainer, TrainingArguments
import torch
from PIL import Image
import requests
from datasets import Dataset
from peft import LoraConfig, get_peft_model

class LlavaLora:
    """
    A class for fine-tuning LLaVA models using LoRA.

    This class provides functionality for loading, configuring, and fine-tuning
    LLaVA (Large Language and Vision Assistant) models using Low-Rank Adaptation (LoRA).

    Attributes:
        model_name (str): Name or path of the pre-trained LLaVA model
        device (torch.device): Device to run the model on (CPU/GPU/MPS)
        processor (LlavaNextProcessor): Processor for handling inputs
        model (LlavaNextForConditionalGeneration): The LLaVA model
        output_dir (str): Directory to save the fine-tuned model
    """

    def __init__(self,
                 model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf",
                 output_dir: str = "./llava-finetuned",
                 device: str = None):
        """
        Initialize the LlavaLoraFt instance.

        Args:
            model_name (str): Name or path of the pre-trained LLaVA model
            output_dir (str): Directory to save the fine-tuned model
            device (str, optional): Specific device to use. If None, will auto-detect
        """
        self.model_name = model_name
        self.output_dir = output_dir

        # Set device
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # Initialize processor and model
        self._initialize_processor()
        self._initialize_model()

    def _initialize_processor(self):
        """
        Initialize the LLaVA processor with specific configurations.
        """
        self.processor = LlavaNextProcessor.from_pretrained(
            self.model_name,
            image_processor_kwargs={
                "size": {"height": 336, "width": 336},  # Set specific image size
                "patch_size": 14,  # Changed from 16 to 14
            },
            truncation=True,
            padding="max_length",
            max_length=512,
            revision="2f7f20bda2e7af8e54438fec01ac5214e9ac6f92"
        )
        self.processor.tokenizer.padding_side = "right"

    def _initialize_model(self):
        """
        Initialize and configure the LLaVA model with LoRA settings.
        """
        # Load base model
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            revision="2f7f20bda2e7af8e54438fec01ac5214e9ac6f92",
            vision_config_kwargs={
                "patch_size": 14,  # Match patch_size with processor
                "image_size": 336,  # Match image_size with processor
            }
        )

        # Set model padding side for training
        self.model.padding_side = "right"

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # Convert to PEFT model
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)

    def prepare_dataset(self, image_paths: list, prompts: list, responses: list) -> Dataset:
        """
        Prepare a dataset from lists of image paths, prompts, and responses.

        Args:
            image_paths (list): List of paths to image files
            prompts (list): List of prompt texts
            responses (list): List of response texts

        Returns:
            Dataset: HuggingFace Dataset object
        """
        dataset_dict = {
            "image": image_paths,
            "prompt": prompts,
            "response": responses
        }
        return Dataset.from_dict(dataset_dict)

    def _preprocess_function(self, examples: dict) -> dict:
        """
        Preprocess the dataset examples for training.
        """
        prompts = examples["prompt"]
        image_paths = examples["image"]
        responses = examples["response"]

        images = []
        for img_path in image_paths:
            try:
                image = Image.open(img_path)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                images.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                raise

        conversations = []
        for prompt, response in zip(prompts, responses):
            conversation = (
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<image>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
            )
            conversations.append(conversation)


        try:
            inputs = self.processor(
                images=images,
                text=conversations,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )
        except Exception as e:
            print("Processor error:")
            print(f"Images type: {type(images)}, length: {len(images)}")
            print(f"First image type: {type(images[0])}")
            print(f"Text type: {type(conversations)}, length: {len(conversations)}")
            print(f"First text: {conversations[0][:100]}...")
            raise

        # 4. 设置标签
        inputs["labels"] = inputs["input_ids"].clone()

        # 5. 清理不需要的字段
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        return inputs

    def _collate_fn(self, examples: list) -> dict:
        """
        Collate function for batching examples during training.

        Args:
            examples (list): List of dictionaries containing prompt, image, and response

        Returns:
            dict: Processed batch ready for training
        """
        batch = {
            "prompt": [],
            "image": [],
            "response": []
        }

        for example in examples:
            batch["prompt"].append(example["prompt"])
            batch["image"].append(example["image"])
            batch["response"].append(example["response"])

        return self._preprocess_function(batch)

    def train(self,
              train_dataset: Dataset,
              num_epochs: int = 3,
              batch_size: int = 1,
              gradient_accumulation_steps: int = 4,
              learning_rate: float = 2e-4,
              save_steps: int = 100,
              logging_steps: int = 10):
        """
        Train the model using the provided dataset.

        Args:
            train_dataset (Dataset): Dataset to use for training
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            gradient_accumulation_steps (int): Number of steps to accumulate gradients
            learning_rate (float): Learning rate for training
            save_steps (int): Save checkpoint every N steps
            logging_steps (int): Log metrics every N steps
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=False,
            save_steps=save_steps,
            logging_steps=logging_steps,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=self._collate_fn,
        )

        print("Starting training...")
        trainer.train()

        # Save the fine-tuned model
        self.save_model()

    def save_model(self):
        """
        Save the fine-tuned model to the specified output directory.
        """
        self.model.save_pretrained(self.output_dir)