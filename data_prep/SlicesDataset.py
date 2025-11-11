"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset
import numpy as np
#import torchvision import transforms 
    
class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data, transform, p_transform=None):
        
        self.data = data
        self.transform = transform
        self.p_transform = p_transform
        
        self.slices = []
        self.img_2d = []
        self.seg_2d = []
        self.file_name = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))
        # for i, d in enumerate(data):
            for img in d['image']:
                self.img_2d.append(img)
            for seg in d['seg']:
                self.seg_2d.append(seg)
                

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc   = self.slices[idx]
        img2d = self.img_2d[idx]
        seg2d = self.seg_2d[idx]

        # You could implement caching strategy here if dataset is too large to fit
        # in memory entirely
        # Also this would be the place to call transforms if data augmentation is used
        
        # TASK: Create two new keys in the "sample" dictionary, named "image" and "seg"
        # The values are 3D Torch Tensors with image and label data respectively. 
        # First dimension is size 1, and last two hold the voxel data from the respective
        # slices. Write code that stores the 2D slice data in the last 2 dimensions of the 3D Tensors. 
        # Your tensor needs to be of shape [1, patch_size, patch_size]
        # Don't forget that you need to put a Torch Tensor into your dictionary element's value
        # Hint: your 3D data sits in self.data variable, the id of the 3D volume from data array
        # and the slice number are in the slc variable. 
        # Hint2: You can use None notation like so: arr[None, :] to add size-1 
        # dimension to a Numpy array
        # <YOUR CODE GOES HERE>
        # reshape images and segs        
        
        i2 = np.reshape(img2d, newshape=(1,img2d.shape[0],img2d.shape[1]))        
        s2 = np.reshape(seg2d, newshape=(1,seg2d.shape[0],seg2d.shape[1]))

        i2 = torch.from_numpy(i2).to(device='cpu',dtype=torch.float)
        s2 = torch.from_numpy(s2).to(device='cpu',dtype=torch.long)
        
        # normalize first - no need for normalization in this case
        # i2 = self.p_normalize(i2)
        
        # then transform
        i2, s2 = self.transform(i2, s2)
        
        # img2d_cuda = torch.from_numpy(i2).to(device='cuda',dtype=torch.float)
        # seg2d_cuda = torch.from_numpy(s2).to(device='cuda',dtype=torch.long)
        
        img2d_cuda = i2.to(device='cuda',dtype=torch.float)
        seg2d_cuda = s2.to(device='cuda',dtype=torch.long)
        
        slice_data_2d = {"indices": slc, "images": img2d_cuda, "segs": seg2d_cuda}
        
        return slice_data_2d

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)
