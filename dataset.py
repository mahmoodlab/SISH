import h5py
import torch
import numpy as np
from tqdm import tqdm


class WSIPatchDataset(torch.utils.data.Dataset):
    def __init__(self, input_file):
        self.input_file = input_file
        imgs_tmp = []
        print("Loading data...")
        with open(self.input_file, 'r') as handle:
            for file_path in tqdm(handle.readlines()):
                file_path = file_path.strip()
                with h5py.File(file_path, 'r') as hf:
                    imgs_tmp.append(torch.from_numpy(np.array(hf['imgs'])).permute(0, 3, 2, 1))
        self.imgs = torch.cat(imgs_tmp, axis=0)

    def _transforms(self, x):
        x_tensor = x / 255.
        return 2 * x_tensor - 1

    def __getitem__(self, index):
        return self._transforms(self.imgs[index])

    def __len__(self):
        return len(self.imgs)


class Mosaic_Bag_FP(torch.utils.data.Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 resolution,
                 custom_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            wsi (openslide object): Whole slide image loaded by openslide
            resolution (int): The resolution of the wsi
            custom_transforms (callable, optional): The transform to be applied on a sample
        """
        self.wsi = wsi
        self.resolution = resolution
        self.roi_transforms = custom_transforms
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            f = h5py.File(self.file_path, "r")
            self.dset = f['coords'][:]

        self.patch_level = 0
        if self.resolution == 40:
            self.patch_size = 2048  # 512
            self.target_patch_size = 1024  # 256
        elif self.resolution == 20:
            self.patch_size = 1024  # 256
            self.target_patch_size = 1024  # 256
        self.length = len(self.dset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = self.wsi.read_region((self.dset[idx][0], self.dset[idx][1]),
                                   self.patch_level,
                                   (self.patch_size, self.patch_size)).convert('RGB')
        img = img.resize((self.target_patch_size, self.target_patch_size))
        img = self.roi_transforms(img)
        return img, self.dset[idx]
