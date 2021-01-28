import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch

class CustomDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A1 = os.path.join(opt.dataroot, opt.phase + 'A1')  # create a path '/path/to/data/trainA1'
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + 'A2')  # create a path '/path/to/data/trainA2'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A1_paths = sorted(make_dataset(self.dir_A1, opt.max_dataset_size))   # load images from '/path/to/data/trainA1'
        self.A2_paths = sorted(make_dataset(self.dir_A2, opt.max_dataset_size))   # load images from '/path/to/data/trainA2'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A1_size = len(self.A1_paths)  # get the size of dataset A
        self.A2_size = len(self.A2_paths)
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A1_path = self.A1_paths[index % self.A1_size]  # make sure index is within then range
        A2_path = self.A2_paths[index % self.A2_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A1_path).convert('RGBA')
        A2_img = Image.open(A2_path).convert('RGBA')
        B_img = Image.open(B_path).convert('RGB')
        A_img.save('trainA1_img', 'PNG')
        A2_img.save('trainA2_img', 'PNG')
        #A_img.paste(A2_img, (0,0), A2_img)
        #B_img.paste(A2_img, (0,0), A2_img)
        A_img = Image.blend(A_img, A2_img, alpha=0.3)
        A_img.save('trainAblend', 'PNG')
        #Ai = self.blend_images(A_img, A2_img)
        #Bi = self.blend_images(B_img, A2_img)
        
        
        # apply image transformation
        A = self.transform_A(A_img.convert('RGB'))
        B = self.transform_B(B_img)
        


        return {'A': A, 'B': B, 'A_paths': A1_path, 'B_paths': B_path}

    def blend_images(self, im1, im2):
        newimg1 = Image.new('RGBA', size=(64, 64), color=(0,0,0,0))
        newimg1.paste(im1, (0,0))
        newimg1.paste(im2, (0,0))
        
        newimg2 = Image.new('RGBA', size=(64, 64), color=(0,0,0,0))
        newimg2.paste(im2, (0,0))
        newimg2.paste(im1, (0,0))
        
        return Image.blend(newimg1, newimg2, alpha=0.9).convert('RGB')
        

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A1_size, self.A2_size, self.B_size)
