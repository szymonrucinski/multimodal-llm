import torch
import numpy as np
from torch.nn import functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoModelForCausalLM, CLIPVisionModel, AutoTokenizer
from class_Projection import PatchReducerWithProjections

clip_model_name = "openai/clip-vit-base-patch32"
phi_model_name = "microsoft/phi-2"


class CLIPPhi2Model(LightningModule):
    def __init__(self, clip_embed=768, phi_embed=2560):
        super().__init__()

        self.EOS_TOKEN_ID = 50256
        self.IMAGE_TOKEN_ID = 50293

        self.tokenizer = AutoTokenizer.from_pretrained(
            phi_model_name, trust_remote_code=True
        )
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        self.phi2_model = AutoModelForCausalLM.from_pretrained(
            phi_model_name, torch_dtype=torch.float32, trust_remote_code=True
        )

        self.projection = PatchReducerWithProjections(49, 30, clip_embed, phi_embed)

        # Freeze Weights
        for network in [self.phi2_model, self.clip_model]:
            for param in network.parameters():
                param.requires_grad_(False)

    def forward(self, images, target_captions, use_teacher_forcing):
        batch_size = target_captions.size(0)
        target_length = target_captions.shape[1]

        clip_outputs = self.clip_model(**images)
        images = clip_outputs.last_hidden_state[:, 1:, :]

        image_embeds = self.projection(images)

        num_patches = image_embeds.size(1)
        image_embeds_flat = image_embeds.view(batch_size, num_patches, -1)

        image_tokenized = torch.tensor(self.IMAGE_TOKEN_ID).repeat(batch_size, 1)
        image_token_embed = self.phi2_model.model.embed_tokens(
            image_tokenized.to(image_embeds.device)
        )

        combined_embeds = torch.zeros(
            batch_size,
            target_length + num_patches + 1,
            image_embeds.shape[-1],
            device=image_embeds.device,
        )

        combined_embeds[:, :num_patches, :].copy_(image_embeds_flat)
        combined_embeds[:, num_patches, :].copy_(image_token_embed.squeeze(1))

        logits_history = torch.empty(
            (batch_size, target_length - 1, self.phi2_model.config.vocab_size),
            device=combined_embeds.device,
        )

        for t in range(num_patches + 1, target_length + num_patches):
            combined_embeds_slice = combined_embeds[:, :t, :].clone()

            outputs = self.phi2_model(inputs_embeds=combined_embeds_slice)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            next_step_logits = logits[:, -1, :].unsqueeze(1)
            logits_history[:, t - (num_patches + 1), :] = next_step_logits.squeeze(1)

            sampling_rate = scheduled_sampling_rate(
                self.global_step, self.trainer.estimated_stepping_batches
            )
            if use_teacher_forcing and torch.rand(1).item() <= sampling_rate:
                next_token = target_captions[:, t - (num_patches + 1)]
            else:
                next_token = torch.argmax(next_step_logits, dim=-1)

            next_token_embeds = self.phi2_model.model.embed_tokens(next_token)
            combined_embeds[:, t, :].copy_(next_token_embeds.squeeze(1))

        del combined_embeds
        del clip_outputs
        del image_embeds
        del image_embeds_flat
        del next_token_embeds
        del images
        # torch.cuda.empty_cache()
        return logits_history

    def common_step(self, batch, use_teacher_forcing):
        images, target_captions = batch
        images = {"pixel_values": images}
        outputs = self.forward(images, target_captions, use_teacher_forcing)
        loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            target_captions[:, 1:].contiguous().view(-1),
            ignore_index=self.EOS_TOKEN_ID,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, use_teacher_forcing=True)
        self.log("train/loss", loss.item(), prog_bar=True, on_step=True, logger=True)

        if self.global_step % 400 == 0:
            self.evaluate_model_performance(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, use_teacher_forcing=False)
        self.log(
            "val/loss",
            loss.item(),
            prog_bar=True,
            on_step=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4
        )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-3,
            pct_start=0.05,
            total_steps=self.trainer.max_steps,
            anneal_strategy="cos",
            div_factor=100,
            final_div_factor=1000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):

        keys = checkpoint["state_dict"].keys()
        keys = [k for k in keys if "projection" not in k]

        for k in keys:
            del checkpoint["state_dict"][k]

    def evaluate_model_performance(self, batch):
        images, target_captions = batch
        images = {"pixel_values": images}
        batch_size = target_captions.size(0)

        # Forward pass through the model
        logits = self.forward(images, target_captions, use_teacher_forcing=False)
        predictions = torch.argmax(logits, dim=-1)

        # Iterate over each item in the batch
        for i in range(batch_size):
            print(f"Image {i+1}:")

            # Ground Truth
            ground_truth_caption = self.tokenizer.decode(
                target_captions[i], skip_special_tokens=True
            )
            print(f"Ground Truth: {ground_truth_caption}")

            # Model Prediction
            predicted_caption = self.tokenizer.decode(
                predictions[i], skip_special_tokens=True
            )
            print(f"Predicted: {predicted_caption}")
            print("---------------------------------------------------")


def scheduled_sampling_rate(
    epoch, max_epochs, start_rate=1.0, end_rate=0.0, decay="exponential"
):
    if decay == "linear":
        return start_rate - (start_rate - end_rate) * (epoch / max_epochs)
    elif decay == "exponential":
        return start_rate * (end_rate / start_rate) ** (epoch / max_epochs)
    elif decay == "inverse_sigmoid":
        return end_rate + (start_rate - end_rate) * (
            max_epochs / (max_epochs + np.exp(epoch / max_epochs))
        )
