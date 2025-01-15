import torch
from PIL import Image
from diffusion_model.blip_override.blip import init_tokenizer
from transformers import CLIPTokenizer
from torch.utils.data import Dataset
from omegaconf import DictConfig
from torchvision import transforms


class VisDroneDataset(Dataset):
    def __init__(self, dataframe, args: DictConfig):
        self.args = args
        self.dataframe = dataframe
        self.image_paths = dataframe['image_path'].tolist()

        self.transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.max_length = args.VisDrone.max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()

        vector = self.clip_tokenizer.add_tokens(list(args.VisDrone.new_tokens))
        print("Clip {} new tokens added".format(vector))
        vector = self.blip_tokenizer.add_tokens(list(args.VisDrone.new_tokens))
        print("Blip {} new tokens added".format(vector))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        path = self.image_paths[idx]
        image = Image.open(image_path)
        description = self.dataframe.iloc[idx, 2]
        annotations = self.dataframe.iloc[idx, 3]

        # Reigon format <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height> -> (x_min, y_min, width, height)
        boxes = [
            [anno[0], anno[1], anno[0] + max(anno[2], 1), anno[1] + max(anno[3], 1)]
            for anno in annotations
        ]

        labels = [anno[5] for anno in annotations]

        width, height = image.size

        # Scaling factors for width and height
        scale_x = 1024 / width
        scale_y = 1024 / height

        # Scale the bounding boxes
        scaled_boxes = [
            [box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y]
            for box in boxes
        ]

        targets = {
            'boxes': torch.as_tensor(scaled_boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }

        tokenized = self.clip_tokenizer(
            description,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        # Clip Model
        msg, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        tokenized = self.blip_tokenizer(
            description,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        # Blip Model
        src_msg, src_attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        if self.transform:
            image = self.transform(image)

        return image, path, targets,  msg, src_msg, attention_mask, src_attention_mask
