# /home/ponsankar/mde/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class AutonomousVehicleDataset(Dataset):
    def __init__(self, data_dir, mode="val", task="depth_prediction", transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.task = task
        self.transform = transform
        self.image_dir = os.path.join(data_dir, "image")
        self.intrinsics_dir = os.path.join(data_dir, "intrinsics")
        self.velodyne_dir = os.path.join(data_dir, "velodyne_raw") if task == "depth_completion" else None
        self.gt_dir = os.path.join(data_dir, "groundtruth_depth") if mode == "val" else None
        self.image_files = sorted(os.listdir(self.image_dir)) if os.path.exists(self.image_dir) else []
        if self.gt_dir and os.path.exists(self.gt_dir):
            self.gt_files = sorted(os.listdir(self.gt_dir))
        if self.velodyne_dir and os.path.exists(self.velodyne_dir):
            self.velodyne_files = sorted(os.listdir(self.velodyne_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        intrinsics_path = os.path.join(self.intrinsics_dir, self.image_files[idx].replace(".png", ".txt"))
        intrinsics = np.loadtxt(intrinsics_path).reshape(3, 3) if os.path.exists(intrinsics_path) else np.eye(3)
        sparse_depth = None
        if self.velodyne_dir and self.task == "depth_completion":
            velodyne_path = os.path.join(self.velodyne_dir, self.image_files[idx])
            sparse_depth = np.array(Image.open(velodyne_path)).astype(np.float32) / 256.0
        depth_gt = None
        if self.mode == "val" and self.gt_dir:
            gt_path = os.path.join(self.gt_dir, self.gt_files[idx])
            depth_gt = np.array(Image.open(gt_path)).astype(np.float32) / 256.0
        if self.transform:
            pass
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        intrinsics = torch.from_numpy(intrinsics).float()
        if sparse_depth is not None:
            sparse_depth = torch.from_numpy(sparse_depth).float().unsqueeze(0)
        if depth_gt is not None:
            depth_gt = torch.from_numpy(depth_gt).float().unsqueeze(0)
        sample = {"image": image, "intrinsics": intrinsics}
        if sparse_depth is not None:
            sample["sparse_depth"] = sparse_depth
        if depth_gt is not None:
            sample["depth_gt"] = depth_gt
        return sample