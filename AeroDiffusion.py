import inspect
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, AutoencoderKL, LMSDiscreteScheduler, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from diffusion_model.blip_override.blip import blip_feature_extractor, init_tokenizer
from diffusion_model.diffusers_override.unet_2d_condition import UNet2DConditionModel
from omegaconf import DictConfig
import pytorch_lightning as pl
from diffusers.optimization import get_cosine_schedule_with_warmup

from region_transformer import RegionTransformer


class DiffusionModel(pl.LightningModule):
    # noinspection Annotator
    def __init__(self, args: DictConfig, steps_per_epoch=1):
        super(DiffusionModel, self).__init__()

        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.task = args.task
        self.reigon_emb = RegionTransformer()

        if args.mode == 'test':

            if args.scheduler == 'ddim':
                self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               clip_sample=False, set_alpha_to_one=True)
            elif args.scheduler == 'ddpm':
                self.scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                               num_train_timesteps=1000)
            else:
                raise ValueError('Scheduler not supported')

        # Unet model
        self.unet = UNet2DConditionModel.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='unet')
        # Auto Encoder
        self.vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="vae")
        # Clip Text Model
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='tokenizer')
        # BLIP Model
        self.blip_tokenizer = init_tokenizer()
        self.blip_image_processor = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.max_length = args.VisDrone.max_length

        # Initializing the Null tokens for the model
        blip_image_null_token = self.blip_image_processor(
            Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))).unsqueeze(0).float()

        clip_text_null_token = self.clip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids
        blip_text_null_token = self.blip_tokenizer([""], padding="max_length", max_length=self.max_length,
                                                   return_tensors="pt").input_ids

        self.register_buffer('clip_text_null_token', clip_text_null_token)
        self.register_buffer('blip_text_null_token', blip_text_null_token)
        self.register_buffer('blip_image_null_token', blip_image_null_token)

        self.text_encoder = CLIPTextModel.from_pretrained('runwayml/stable-diffusion-v1-5',
                                                          subfolder='text_encoder')
        self.text_encoder.resize_token_embeddings(args.VisDrone.clip_embedding_tokens)

        # Resizing Position Embedding
        old_embeddings = self.text_encoder.text_model.embeddings.position_embedding
        new_embeddings = self.text_encoder._get_resized_embeddings(old_embeddings, self.args.VisDrone.max_length)
        self.text_encoder.text_model.embeddings.position_embedding = new_embeddings
        self.text_encoder.config.max_position_embeddings = self.max_length
        self.text_encoder.max_position_embeddings = self.max_length
        self.text_encoder.text_model.embeddings.position_ids = torch.arange(self.args.VisDrone.max_length).expand(
            (1, -1))

        # Time and Type embeddings
        self.modal_type_embeddings = nn.Embedding(2, 768)
        self.time_embeddings = nn.Embedding(5, 768)

        # Multi-Modal Embeddings
        self.mm_encoder = blip_feature_extractor(
            pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth',
            image_size=512, vit='large')
        self.mm_encoder.text_encoder.resize_token_embeddings(args.VisDrone.blip_embedding_tokens)

        self.region_mm_encoder = blip_feature_extractor(
            pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large.pth',
            image_size=224, vit='large')
        self.region_mm_encoder.text_encoder.resize_token_embeddings(args.VisDrone.blip_embedding_tokens)

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                             num_train_timesteps=1000)
        # Freeze vae and unet
        self.freeze_params(self.vae.parameters())
        if args.freeze_resnet:
            self.freeze_params([p for n, p in self.unet.named_parameters() if "attentions" not in n])

        if args.freeze_blip and hasattr(self, "mm_encoder"):
            self.freeze_params(self.mm_encoder.parameters())
            self.unfreeze_params(self.mm_encoder.text_encoder.embeddings.word_embeddings.parameters())

        if args.freeze_blip and hasattr(self, "region_mm_encoder"):
            self.freeze_params(self.region_mm_encoder.parameters())
            self.unfreeze_params(self.region_mm_encoder.text_encoder.embeddings.word_embeddings.parameters())

        if args.freeze_clip and hasattr(self, "text_encoder"):
            self.freeze_params(self.text_encoder.parameters())
            self.unfreeze_params(self.text_encoder.text_model.embeddings.token_embedding.parameters())

    @staticmethod
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    @staticmethod
    def unfreeze_params(params):
        for param in params:
            param.requires_grad = True

    def forward(self, batch):
        if self.args.freeze_clip and hasattr(self, "text_encoder"):
            self.text_encoder.eval()
        if self.args.freeze_blip and hasattr(self, "mm_encoder"):
            self.mm_encoder.eval()

        images, image_path, targets, msg, src_msg, attention_mask, src_attention_mask = batch

        B, V, S = msg.shape
        src_V = V + 1 if self.task == 'continuation' else V

        captions = torch.flatten(msg, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        source_caption = torch.flatten(src_msg, 0, 1)
        source_attention_mask = torch.flatten(src_attention_mask, 0, 1)
        # 1 is not masked, 0 is masked

        image_path = f"{image_path[0]}"
        image_regions, region_input_ids, region_attention_mask = self.reigon_emb(image_path, self.device)

        classifier_free_idx = np.random.rand(B * V) < 0.1
        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state  # B * V, S, D

        source_embeddings = self.mm_encoder(images, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)


        region_embeddings = self.region_mm_encoder(image_regions, region_input_ids, region_attention_mask,
                                                   mode='multimodal').reshape(B, src_V * S, -1)
        projection_layer = nn.Linear(region_embeddings.size(2), 768).to(self.device)
        # Project the embeddings to a lower dimension
        region_embeddings = projection_layer(region_embeddings)
        region_embeddings = region_embeddings.repeat_interleave(V, dim=0)
        region_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))

        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)
        caption_embeddings[classifier_free_idx] = \
            self.text_encoder(self.clip_text_null_token).last_hidden_state[0]
        source_embeddings[classifier_free_idx] = \
            self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token, attention_mask=None,
                            mode='multimodal')[0].repeat(src_V, 1)

        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))

        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings, region_embeddings], dim=1)

        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)


        additional_attention_mask, _ = torch.max(region_embeddings, dim=-1)
        attention_mask = torch.cat([attention_mask, additional_attention_mask], dim=1)

        attention_mask = ~(attention_mask.bool())  # B * V, (src_V + 1) * S
        attention_mask[classifier_free_idx] = False

        # B, V, V, S
        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215

        noise = torch.randn(latents.shape, device=self.device)
        bsz = latents.shape[0]

        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, attention_mask).sample
        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

        return loss

    def sample(self, batch):

        images, image_path, targets, msg, src_msg, attention_mask, src_attention_mask = batch
        B, V, S = msg.shape
        src_V = V + 1 if self.task == 'continuation' else V

        image_filename = os.path.basename(image_path[0])
        image_path = f"{image_path[0]}"

        captions = torch.flatten(msg, 0, 1)
        attention_mask = torch.flatten(attention_mask, 0, 1)
        source_caption = torch.flatten(src_msg, 0, 1)
        source_attention_mask = torch.flatten(src_attention_mask, 0, 1)

        caption_embeddings = self.text_encoder(captions, attention_mask).last_hidden_state  # B * V, S, D
        source_embeddings = self.mm_encoder(images, source_caption, source_attention_mask,
                                            mode='multimodal').reshape(B, src_V * S, -1)
        caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        source_embeddings = source_embeddings.repeat_interleave(V, dim=0)


        image_regions, region_input_ids, region_attention_mask = self.reigon_emb(image_path, self.device)

        region_embeddings = self.region_mm_encoder(image_regions, region_input_ids, region_attention_mask,
                                                   mode='multimodal').reshape(B, src_V * S, -1)

        projection_layer = nn.Linear(region_embeddings.size(2), 768).to(self.device)

        region_embeddings = projection_layer(region_embeddings)
        region_embeddings = region_embeddings.repeat_interleave(V, dim=0)
        region_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))

        encoder_hidden_states = torch.cat([caption_embeddings, source_embeddings, region_embeddings], dim=1)

        attention_mask = torch.cat(
            [attention_mask, source_attention_mask.reshape(B, src_V * S).repeat_interleave(V, dim=0)], dim=1)

        additional_attention_mask, _ = torch.max(region_embeddings, dim=-1)
        attention_mask = torch.cat([attention_mask, additional_attention_mask], dim=1)

        attention_mask = ~(attention_mask.bool())  # B * V, (src_V + 1) * S

        # B, V, V, S
        square_mask = torch.triu(torch.ones((V, V), device=self.device)).bool()
        square_mask = square_mask.unsqueeze(0).unsqueeze(-1).expand(B, V, V, S)
        square_mask = square_mask.reshape(B * V, V * S)
        attention_mask[:, -V * S:] = torch.logical_or(square_mask, attention_mask[:, -V * S:])

        uncond_caption_embeddings = self.text_encoder(self.clip_text_null_token).last_hidden_state
        uncond_source_embeddings = self.mm_encoder(self.blip_image_null_token, self.blip_text_null_token,
                                                   attention_mask=None, mode='multimodal').repeat(1, src_V, 1)
        uncond_caption_embeddings += self.modal_type_embeddings(torch.tensor(0, device=self.device))
        uncond_source_embeddings += self.modal_type_embeddings(torch.tensor(1, device=self.device))
        uncond_source_embeddings += self.time_embeddings(
            torch.arange(src_V, device=self.device).repeat_interleave(S, dim=0))
        uncond_embeddings = torch.cat([uncond_caption_embeddings, uncond_source_embeddings], dim=1)
        uncond_embeddings = uncond_embeddings.expand(B * V, -1, -1)

        uncond_embeddings = torch.cat(
            [uncond_embeddings, torch.zeros((1, 200, uncond_embeddings.size(2)), device=self.device)], dim=1)

        encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])
        uncond_attention_mask = torch.zeros((B * V, (src_V + 1) * S), device=self.device).bool()

        additional_embeddings_mask = torch.zeros((1, 200), device=self.device).bool()
        uncond_attention_mask = torch.cat([uncond_attention_mask, additional_embeddings_mask], dim=1)

        uncond_attention_mask[:, -V * S:] = square_mask
        attention_mask = torch.cat([uncond_attention_mask, attention_mask], dim=0)

        attention_mask = attention_mask.reshape(2 * B, V, ((V + 1) * S) + 200)

        generated_images = list()
        for i in range(V):
            encoder_hidden_states = encoder_hidden_states.reshape(2, B, V, ((V + 1) * S) + 200, -1)

            new_image = self.diffusion(encoder_hidden_states.reshape(2 * B, ((V + 1) * S) + 200, -1),
                                       attention_mask.reshape(2 * B, ((V + 1) * S) + 200),
                                       512, 512, self.args.num_inference_steps, self.args.guidance_scale, 0.0)

            generated_images += new_image

        return images, generated_images, image_filename

    def diffusion(self, encoder_hidden_states, attention_mask, H, W, num_inference_steps, guidance_scale, eta):

        latents = torch.randn((encoder_hidden_states.shape[0] // 2, self.unet.in_channels, H // 8, W // 8),
                              device=self.device)
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents * self.scheduler.sigmas[0]

        # set time step
        accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {}
        if accepts_offset:
            extra_set_kwargs["offset"] = 1

        self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        extra_step_kwargs = {"eta": eta} if "eta" in inspect.signature(self.scheduler.step).parameters else {}
        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states, attention_mask).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # Scale and decode the latent images
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        return self.numpy_to_pil(image)

    @staticmethod
    def numpy_to_pil(images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image, 'RGB') for image in images]

        return pil_images

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=1e-4)
        # Calculate the total number of training steps
        num_training_steps = self.args.max_epochs * self.steps_per_epoch

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_epochs * self.steps_per_epoch,
            num_training_steps=num_training_steps,
            num_cycles=0.5  # This defines the shape of the cosine curve, 0.5 for half a cosine
        )

        optim_dict = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,  # The updated LR scheduler instance
                'interval': 'step',  # Changed to 'step' as `get_cosine_schedule_with_warmup` updates every step
            }
        }
        return optim_dict

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('loss/train_loss', loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('epoch', self.current_epoch, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        orig_image, generated_images, image_filename = self.sample(batch)
        save_path = 'path/to/save/original/images'
        for i, image in enumerate(orig_image):
            filename = os.path.join(save_path, f'{image_filename}')
            vutils.save_image(image, filename)

        return generated_images
