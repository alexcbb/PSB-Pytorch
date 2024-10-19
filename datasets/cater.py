import lightning as L
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import os
import cv2
from PIL import Image

class CaterDataset(Dataset):
    def __init__(
        self,
        data_dir="./data/",
        max_video_len=301,
        num_sample_frame=6,
        frame_offset=1, 
    ):
        super().__init__()
        all_videos = os.listdir(data_dir)
        self.files = [os.path.join(data_dir, video) for video in all_videos]
        self.num_sample_frame = num_sample_frame
        self.max_video_len = max_video_len
        self.frame_offset = frame_offset
        self.valid_idx = self._get_sample_idx()
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
        ])
        
    def _get_sample_idx(self):
        """Get (video_idx, start_frame_idx) pairs as a list."""
        valid_idx = []
        for video_idx in range(len(self.files)):
            # simply use random uniform sampling
            max_start_idx = self.max_video_len - (self.num_sample_frame -1) * self.frame_offset
            valid_idx += [(video_idx, idx) for idx in range(1, max_start_idx)]
        return valid_idx

    def __len__(self):
        return len(self.valid_idx)
    
    def _read_frames(self, idx):
        """Read video frames. Directly read from jpg images."""
        video_idx, start_idx = self.valid_idx[idx]
        video_path = self.files[video_idx]

        # wrong video length !
        if len(os.listdir(video_path)) != self.max_video_len:
            raise ValueError
        frames = [
            cv2.imread(
                os.path.join(
                    video_path,
                    f'frame_{(start_idx + n * self.frame_offset):04d}.jpg'))
            for n in range(self.num_sample_frame)
        ]
        if any(frame is None for frame in frames):
            raise ValueError
        frames = [
            self.transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
            for img in frames
        ]  # [T, C, H, W]
        return torch.stack(frames, dim=0).float()

    def __getitem__(self, idx):
        return {
            "video": self._read_frames(idx)
        }

class CaterDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_dataset = CaterDataset(
            data_dir=self.cfg.data_dir,
            max_video_len=self.cfg.max_video_len,
            num_sample_frame=self.cfg.num_sample_frame,
            frame_offset=self.cfg.frame_offset,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
