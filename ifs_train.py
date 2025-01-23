from ifs_model.ifs_llava_lora import LlavaLora
from ifs_model.ifs_annotation_data import IFSAnnotationData


# Initialize the LlavaLoraFt instance
llava_ft = LlavaLora(
    model_name="llava-hf/llava-v1.6-mistral-7b-hf",
    output_dir="./llava-finetuned"
)

# Prepare your training data
train_images = ["src/example_food.jpg"]
train_prompts = ["What is shown in this image?"]
train_responses = ["This is a pizza, with a source on top. It's caloric value is 100 calories."]

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

# Load the saved model for inference
inference_model = LlavaLora.load_model(save_directory)

# Generate response
response = inference_model.generate_response(test_image_path, test_prompt)
print(f"Model response: {response}")

