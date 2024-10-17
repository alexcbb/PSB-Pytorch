import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils


from .utils import conv_norm_act, deconv_norm_act, deconv_out_shape
from .layers import SoftPositionEmbed, GridProjection, PSBBlock
import lightning as L

class PSB(nn.Module):
    def __init__(
        self,
        input_shape,
        output_size,
        num_slots=8,
        slot_size=192,
        slot_hidden_size=128,
        num_layers=2,
        time_heads=4,
        obj_heads=4,
        resolution=(64, 64),
        enc_channels = [3, 64, 64, 64, 64],
    ):
        super().__init__()
        ## ENCODER : CNN + MLP proj (as SAVi)
        self.resolution = resolution
        if self.resolution[0] > 64:
            self.visual_resolution = tuple(i // 2 for i in self.resolution)
            downsample = True
        else:
            self.visual_resolution = self.resolution
            downsample = False
        self.visual_channels = enc_channels[-1]  # CNN out visual channels
        enc_layers = len(enc_channels) - 1
        self.encoder = nn.Sequential(*[
            conv_norm_act(
                enc_channels[i],
                enc_channels[i + 1],
                kernel_size=5,
                # 2x downsampling for 128x128 image
                stride=2 if (i == 0 and downsample) else 1,
                norm='',
                act='relu' if i != (enc_layers - 1) else '',
            ) for i in range(enc_layers)
        ])  # relu except for the last layer
        self.projection_layer = GridProjection(self.visual_resolution , self.visual_channels, output_size)

        ## PSB-Layer
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.slot_size = slot_size
        psb_layer = PSBBlock(
            embed_dim=output_size,
            ffn_dim=slot_hidden_size,
            inverse_nh=1,
            time_nh=time_heads,
            obj_nh=obj_heads,
            num_slots=self.num_slots,
        )
        self.psb = nn.Sequential(*[psb_layer for _ in range(self.num_layers)])

        ## DECODER: Spatial-Broadcast Decoder (SBD)
        modules = []
        self.dec_resolution = (8, 8)
        dec_channels = [slot_size, 64, 64, 64, 64]
        stride = 2
        out_size = self.dec_resolution[0]
        for i in range(len(dec_channels) - 1):
            if out_size == input_shape[-1]:
                stride = 1
            modules.append(
                deconv_norm_act(
                    dec_channels[i],
                    dec_channels[i + 1],
                    kernel_size=5,
                    stride=stride,
                    norm='',
                    act='relu'))
            out_size = deconv_out_shape(out_size, stride, 5 // 2,
                                        5, stride - 1)
        modules.append(
            nn.Conv2d(dec_channels[-1], 4, kernel_size=1, stride=1, padding=0)
        )
        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.slot_size, self.dec_resolution)
        self.loss_function = nn.MSELoss()


    def encode(self, img, prev_slots=None, lang=None):
        unflatten = False 
        if img.dim() == 5:
            unflatten = True
            B, T, C, H, W = img.shape
            img = img.flatten(0, 1)
        else:
            B, C, H, W = img.shape
            T = 1
        if lang is not None:
            h = self.encoder(img, lang)
        else:
            h = self.encoder(img)
        h = self.projection_layer(h)

        if unflatten:
            h = h.unflatten(0, (B, T)) 
            img = img.unflatten(0, (B, T))
        else:
            h = h.unsqueeze(1)
            img = img.unsqueeze(1)

        # Extract slots
        slots = self.psb(h, prev_slots)
        
        return slots

    def forward(self, img, prev_slots=None, lang=None, train=True):
        B, T = img.shape[:2]
        
        slots, masks_enc = self.encode(img, prev_slots, lang)

        out_dict = {
            'slots': slots,
            'masks_enc': masks_enc,
        }
        # Decode
        if train:
            recons_full, recons, masks_dec, slots = self.decode(slots)
            loss = self.loss_function(img, recons_full.unflatten(0, (B, T)))
            out_dict['masks_dec'] = masks_dec.unflatten(0, (B, T))
            out_dict['recons'] = recons.unflatten(0, (B, T))
            out_dict['recons_full'] = recons_full.unflatten(0, (B, T))
            out_dict['loss'] = loss
        return out_dict
    
    def decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        # `slots` has shape: [B, self.num_slots, self.slot_size].
        bs, num_slots, slot_size = slots.shape
        height, width = self.input_shape[3:]
        num_channels = 3

        decoder_in = slots.view(bs * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.dec_resolution[0],
                                    self.dec_resolution[1])
        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [B*num_slots, 4, H, W].
        out = out.view(bs, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]  # [B, num_slots, 3, H, W]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
        recon_combined = torch.sum(recons * masks, dim=1)  # [B, 3, H, W]
        return recon_combined, recons, masks, slots

class PSBModule(L.LightningModule):
    def __init__(
            self, 
            model
        ):
        super().__init__()
        self.model = model                
        self.validation_outputs = []

    
    def training_step(self, batch, batch_idx):
        img = batch['video']
        out = self.model(img)
        loss = out['loss']
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img = batch['video']
        with torch.no_grad():
            out = self.model(img)
        loss = out['loss']
        self.log('val_loss', loss)
        return loss
    
    def on_validation_epoch_end(self):
        results = []
        # Select 8 random outputs to visualize
        for output in self.validation_outputs:
            out_dict = output
            recons_full = out_dict["recons_full"][0]
            recons = out_dict["recons"][0]
            masks_dec = out_dict["masks_dec"][0]
            img = out_dict["img"][0]
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
                    recons * masks + (1. - masks),  # each slot
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