from diffusion_model.blip_override.blip import init_tokenizer
from yolov5.region import run
import torch
import torch.nn as nn


class RegionTransformer(nn.Module):
    def __init__(self):
        super(RegionTransformer, self).__init__()
        # Initialize the BLIP tokenizer
        self.blip_tokenizer = init_tokenizer()

        # Object label mapping
        self.object_label = {
            0: "pedestrian",
            1: "people",
            2: "bicycle",
            3: "car",
            4: "van",
            5: "truck",
            6: "tricycle",
            7: "awning-tricycle",
            8: "bus",
            9: "motor"
        }

    def convert_labels(self, labels, device):
        input_ids = []
        attention_masks = []

        for label in labels:
            description = self.object_label[label]  # Corrected to use the object_label dictionary
            tokenized = self.blip_tokenizer(
                description,
                padding="max_length",
                max_length=200,
                truncation=True,  # Set truncation to True for safety
                return_tensors="pt"
            )
            # Move tokenized tensors to the specified device
            input_ids.append(tokenized['input_ids'].squeeze(0).to(device))
            attention_masks.append(tokenized['attention_mask'].squeeze(0).to(device))

        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)

        return input_ids, attention_masks

    def forward(self, path, device):
        # Run object detection to get labels and regions
        labels, regions = run(source=path)

        if regions.numel() == 0:
            print("No reigons found")

        # Check if labels is empty and print the path of the image if it is
        if not labels:  # This checks if labels is an empty list
            print(f"No labels detected for image at path: {path}")

        regions = regions.to(device)  # Move regions to the specified device

        # Convert labels to tokenized input
        input_ids, attention_mask = self.convert_labels(labels, device)

        return regions, input_ids, attention_mask


