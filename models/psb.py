import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import torch.optim as optim
import random
from transformers import (
    get_cosine_schedule_with_warmup,
)

from .encoder import resnet18_savi, make_slot_attention_encoder
from .decoder import make_savi_decoder, SpatialBroadcastDecoder
from .layers import PSBBlock, CoordinatePositionEmbed
import lightning as L

# TODO : comment the code
class PSB(nn.Module):
    def __init__(
        self,
        input_shape,
        encoder_type="resnet18",
        num_slots=8,
        slot_size=64,
        psb_hidden_dim=512,
        inverse_nh=1,
        time_nh=4,
        obj_nh=4,
        num_channels=3,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.resolution = input_shape
        if self.resolution[0] == 128:
            self.visual_resolution = tuple(i // 2 for i in self.resolution)
            feature_multiplier = 1
            downsample = 1
        elif self.resolution[0] == 64:
            self.visual_resolution = self.resolution
            feature_multiplier = 0.5
            downsample = 0
        else:
            raise ValueError(f"Invalid resolution: {self.resolution} needed 128x128 or 64x64")


        ## ENCODER : ResNet-18 or Classical CNN
        if encoder_type == "resnet18":
            self.visual_resolution = tuple(i // 8 for i in self.resolution)
            self.encoder = resnet18_savi()
            self.visual_channels = 512
            self.projection_layer = CoordinatePositionEmbed(self.visual_channels, self.visual_resolution, proj_dim=slot_size)
        elif encoder_type == "cnn":
            self.encoder = make_slot_attention_encoder(
                inp_dim=self.num_channels,
                feature_multiplier=feature_multiplier,
                downsamplings=downsample,
            )
            self.visual_channels = int(64 * feature_multiplier)
            self.projection_layer = CoordinatePositionEmbed(self.visual_channels, self.visual_resolution, proj_dim=slot_size)
        else:
            raise ValueError(f"Invalid encoder type: {encoder_type}")

        ## OBJECT-CENTRIC MODULE : Slot-Attention for Video == SA + Transformer
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.psb = nn.ModuleList(
            [
                PSBBlock(
                    embed_dim=self.slot_size,
                    ffn_dim=psb_hidden_dim,
                    inverse_nh=inverse_nh,
                    time_nh=time_nh,
                    obj_nh=obj_nh,
                    num_slots=num_slots,
                    get_mask=True,
                )
                for _ in range(2)
            ]
        )

        ## DECODER: Spatial-Broadcast Decoder (SBD)
        self.dec_resolution = (8, 8) # TODO: make this a parameter
        savi_decoder = make_savi_decoder(
            inp_dim=self.slot_size,
            feature_multiplier=feature_multiplier,
            upsamplings=4 if downsample else 3 
        )
        self.decoder = SpatialBroadcastDecoder(
            inp_dim=self.slot_size,
            outp_dim=self.num_channels,
            backbone=savi_decoder,
            backbone_dim=int(64*feature_multiplier) if encoder_type == "cnn" else 64,
            initial_size=self.dec_resolution,
            pos_embed=CoordinatePositionEmbed(self.slot_size, self.dec_resolution),
        )

    def encode(self, img):
        unflatten = False 
        if img.dim() == 5:
            unflatten = True
            B, T, C, H, W = img.shape
            img = img.flatten(0, 1)
        else:
            B, C, H, W = img.shape
            T = 1
        h = self.encoder(img)
        h = self.projection_layer(h)
        if unflatten:
            h = h.unflatten(0, (B, T)) 
            img = img.unflatten(0, (B, T))
        else:
            h = h.unsqueeze(1)
            img = img.unsqueeze(1)        
        # Extract slots
        in_dict = {
            "features": h,
            "prev_slots": None,
        }
        for psb in self.psb:
            out_dict = psb(in_dict)
            in_dict["prev_slots"] = out_dict["prev_slots"]
        slots = out_dict["prev_slots"]
        masks = out_dict["masks"]
        
        return slots, masks

    def decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        out_dict = self.decoder(slots)
        recons = out_dict['reconstruction']
        masks = out_dict['masks']
        
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
        if "recon_combined" not in out_dict:
            recon_combined = torch.sum(recons * masks, dim=1) # [B, 3, H, W]
        else:
            recon_combined = out_dict["recon_combined"]
        return recon_combined, recons, masks, slots
    
    def forward(self, img, train=True):
        B, T = img.shape[:2]
        slots, masks_enc = self.encode(img)
        out_dict = {
            'slots': slots,
            'masks_enc': masks_enc,
            'video': img,
        }
        # Decode
        if train:
            slots = slots.flatten(0, 1)
            recons_full, recons, masks_dec, slots = self.decode(slots)
            loss = self.loss_function(img, recons_full.unflatten(0, (B, T)))
            out_dict['masks_dec'] = masks_dec.unflatten(0, (B, T))
            out_dict['recons'] = recons.unflatten(0, (B, T))
            out_dict['recons_full'] = recons_full.unflatten(0, (B, T))
            out_dict['loss'] = loss
            out_dict['mse_loss'] = loss
        return out_dict
    
    def loss_function(self, img, recon_combined):
        """Compute the loss function."""
        loss = F.mse_loss(recon_combined, img, reduction='mean')
        return loss

    def output_shape(self):
        return self.output_shape

class PSBModule(L.LightningModule):
    def __init__(
            self, 
            cfg
        ):
        super().__init__()
        self.cfg = cfg
        self.model = PSB(
            input_shape=[cfg.img_size, cfg.img_size],
            encoder_type=cfg.get("encoder_type", "resnet18"),
            num_slots=cfg.num_slots,
            slot_size=cfg.slot_size,
            psb_hidden_dim=cfg.psb_hidden_dim,
            inverse_nh=cfg.inverse_nh,
            time_nh=cfg.time_nh,
            obj_nh=cfg.obj_nh,
        )
        self.validation_outputs = []
        self.automatic_optimization = False

    def on_training_start(self):
        self.random_idx = random.sample(list(range(2, len(self.trainer.train_dataloaders))), 4)
        if 0 not in self.random_idx:
            self.random_idx.append(0)
    
    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()

        img = batch['video']
        out = self.model(img)
        loss = out['loss']
        self.log('train_loss', loss)

        self.manual_backward(loss)
        self.clip_gradients(optimizer, gradient_clip_val= 0.05, gradient_clip_algorithm='norm')
        optimizer.step()
        scheduler.step()

        return loss
    
    def validation_step(self, batch, batch_idx):
        if batch_idx in self.random_idx:
            img = batch['video']
            with torch.no_grad():
                out = self.model(img)
            loss = out['loss']
            self.log('val_loss', loss)
            self.validation_outputs.append(out)
            return loss
    
    def on_validation_start(self):
        # Get 6 random indices to visualize
        self.random_idx = random.sample(list(range(2, len(self.trainer.val_dataloaders))), 4)
        if 0 not in self.random_idx:
            self.random_idx.append(0)
    
    def on_validation_epoch_end(self):
        results = []
        # Select 8 random outputs to visualize
        for output in self.validation_outputs:
            out_dict = output
            recons_full = out_dict["recons_full"][0]
            recons = out_dict["recons"][0]
            masks_dec = out_dict["masks_dec"][0]
            img = out_dict["video"][0]
            # masks_enc = out_dict["masks_enc"]
            save_video = self._make_video_grid(img, recons_full, recons,
                                            masks_dec)
            results.append(save_video)
        self.validation_outputs.clear()
        self.logger.log_video('val/video', [self._convert_video(results)], {"fps": [10]})

    def to_rgb_from_tensor(self, x):
        """Reverse the Normalize operation in torchvision."""
        return (x * 0.5 + 0.5).clamp(0, 1)

    def _make_video_grid(self, imgs, recon_combined, recons, masks):
        """Make a video of grid images showing slot decomposition."""
        # combine images in a way so we can display all outputs in one grid
        # output rescaled to be between 0 and 1
        out = self.to_rgb_from_tensor(
            torch.cat(
                [
                    imgs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks.unsqueeze(2) + (1. - masks.unsqueeze(2) ),  # each slot
                ],
                dim=1,
            ))  # [T, num_slots+2, 3, H, W]
        # stack the slot decomposition in all frames to a video
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                pad_value=1.,
            ) for i in range(recons.shape[0])
        ])  # [T, 3, H, (num_slots+2)*W]
        return save_video

    def _pad_frame(self, video, target_T):
        """Pad the video to a target length at the end"""
        if video.shape[0] >= target_T:
            return video
        dup_video = torch.stack(
            [video[-1]] * (target_T - video.shape[0]), dim=0)
        return torch.cat([video, dup_video], dim=0)

    def _convert_video(self, video, caption=None):
        max_T = max(v.shape[0] for v in video)
        video = [self._pad_frame(v, max_T) for v in video]
        video = torch.cat(video, dim=2)  # [T, 3, B*H, L*W]
        video = (video * 255.).cpu().numpy().astype(np.uint8)
        return video
    
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad] # only keep trainable parameters
        optimizer = optim.Adam(params=params, lr=self.cfg.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.cfg.decay_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
