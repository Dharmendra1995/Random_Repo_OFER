# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#


import os
import re
from abc import ABC
from functools import reduce
from pathlib import Path
import cv2
import clip
import glob

import loguru
import numpy as np
import torch
import trimesh
import scipy.io
from loguru import logger
from skimage.io import imread
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from src.models.flame import FLAME


class BaseDatasetFlame23(Dataset, ABC):
    def __init__(self, name, config, device, isEval):
        cfg  = config
        self.root_name = cfg.identity_folder_name

        actor_list = os.listdir(self.root_name)
        self.img_list = []
        self.identity_list = []
        self.arcface_list = []
        for i in actor_list:

            actor_name = str(i)
            
            actor_folder = os.path.join(self.root_name, i)
            # Construct the search pattern
            # Use glob patterns for both jpg and png files
            img_pattern_jpg = os.path.join(actor_folder, '*.jpg')
            img_pattern_png = os.path.join(actor_folder, '*.png')

            identity_pattern = os.path.join(actor_folder, 'identity*')
            # Use glob to find all files matching the pattern
            image = glob.glob(img_pattern_jpg) + glob.glob(img_pattern_png)
            identity = glob.glob(identity_pattern)
            arcface = glob.glob(os.path.join(actor_folder, f'{i}.npy*'))



            img_abs_path = image[0]
            identity_abs_path = identity[0]
            arcface_abs_path = arcface[0]

            self.img_list.append(img_abs_path)
            self.identity_list.append(identity_abs_path)
            self.arcface_list.append(arcface_abs_path)

           # Add the expected total_images attribute
        self.total_images = len(self.img_list)
    

        self.flame_folder = 'FLAME23_parameters'
        self.farlmodel, self.farlpreprocess = clip.load("ViT-B/16", device="cpu")
        self.clipmodel, self.clippreprocess = clip.load("ViT-B/32", device="cpu")

        self.dinotransform  = T.Compose([
            T.ToTensor(),
            T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.Normalize(mean=[0.5],std=[0.5]),])


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        images_list = []
        imagesfarl_list = []
        clip_list = []
        dinov2_list = []
        imagenames = []
        arcface_list = []


        image_path= self.img_list[index]
        identity_path = self.identity_list[index]
        arcface_path = self.arcface_list[index]
        image_name = str(image_path).split('/')[-1]

        
        flame = {'shape_params': np.load(identity_path, allow_pickle=True)}

        imagefarl = self.farlpreprocess(Image.open(image_path))

        # Resize the image to 512x512
        img = Image.open(image_path)
        img = img.resize((224, 224), Image.LANCZOS)
        
        # Convert to numpy array and normalize
        image = np.array(img, dtype=np.float32) / 255.0


        # ArcFace image
        arcface = np.load(arcface_path, allow_pickle=True)
        arcface = torch.tensor(arcface, dtype=torch.float32)

        clip_image = self.clippreprocess(Image.open(image_path)).unsqueeze(0)
        dinov2_image = self.dinotransform(Image.open(image_path))[:3].unsqueeze(0)

        images_list.append(image)
        imagesfarl_list.append(imagefarl)
        clip_list.append(clip_image)
        dinov2_list.append(dinov2_image)
        imagenames.append(image_name)
        arcface_list.append(arcface)

        images_array = torch.from_numpy(np.array(images_list)).float()
        imagesfarl_array = torch.stack(imagesfarl_list).float()
        arcface_array = torch.stack(arcface_list).float()
        clip_array = torch.stack(clip_list).float()
        dinov2_array = torch.stack(dinov2_list).float()

        
        return {
            'batchsize': torch.tensor(images_array.shape[0]),
            'image': images_array,
            'farl': imagesfarl_array,
            'clip': clip_array,
            'dinov2': dinov2_array,
            'image_name': image_name,
            'flame': flame,
            'dataset': 'demo',
            'arcface': arcface_array
        }


    # def __getitem__(self, index):
    #     image_path = self.img_list[index]
    #     identity_path = self.identity_list[index]
    #     image_name = str(image_path).split('/')[-1]

    #     # Initialize empty lists
    #     images_list = []
    #     imagesfarl_list = []
    #     clip_list = []
    #     dinov2_list = []
    #     arcface_list = []  # Add this for arcface
    #     imagenames = []

    #     # Load shape parameters

    #     shape_params = np.load(identity_path, allow_pickle=True)
    #     shape_params = torch.tensor(shape_params, dtype=torch.float32)
    #     flame = {'shape_params': shape_params}


    #     # Process image

    #     # Original image
    #     image = np.array(imread(image_path), dtype=np.float32) / 255.0

    #     # FARL image
    #     imagefarl = self.farlpreprocess(Image.open(image_path))

    #     # ArcFace image (using same process as FARL for now)
    #     arcface = self.farlpreprocess(Image.open(image_path))

    #     # CLIP image
    #     clip_image = self.clippreprocess(Image.open(image_path)).unsqueeze(0)

    #     # DINOv2 image
    #     dinov2_image = self.dinotransform(Image.open(image_path))[:3].unsqueeze(0)

    #     # Add to lists
    #     images_list.append(image)
    #     imagesfarl_list.append(imagefarl)
    #     arcface_list.append(arcface)  # Add arcface list
    #     clip_list.append(clip_image)
    #     dinov2_list.append(dinov2_image)
    #     imagenames.append(image_name)


    #     # Convert to tensors with consistent shapes
    #     images_array = torch.from_numpy(np.array(images_list)).float()
    #     imagesfarl_array = torch.stack(imagesfarl_list).float()
    #     arcface_array = torch.stack(arcface_list).float()  # Add arcface array
    #     clip_array = torch.cat(clip_list).float()
    #     dinov2_array = torch.cat(dinov2_list).float()

    #     # Add all required keys with appropriate defaults for missing data
    #     # return {
    #     #     'batchsize': torch.tensor(images_array.shape[0]),
    #     #     'image': images_array,
    #     #     'farl': imagesfarl_array,
    #     #     'arcface': arcface_array,  # Required by encoder
    #     #     'clip': clip_array,
    #     #     'dinov2': dinov2_array,
    #     #     'image_name': image_name,  # Change to match what validator expects
    #     #     'imagename': image_name,   # Keep for backward compatibility
    #     #     'dataset': self.name,      # Add dataset name
    #     #     'flame': flame,
    #     #     # Add other required fields with default values
    #     #     'pose': torch.zeros(1, 3),
    #     #     'pose_valid': torch.zeros(1),
    #     #     'lmk': torch.zeros(1, 68, 2),
    #     #     'lmk_valid': torch.zeros(1),
    #     #     'exp': torch.zeros(1)
    #     # }


    #     return {
    #         'batchsize': torch.tensor(images_array.shape[0]),
    #         'image': images_array,
    #         'farl': imagesfarl_array,
    #         'arcface': imagesfarl_array,
    #         'clip': clip_array,
    #         'dinov2': dinov2_array,
    #         'pose': dinov2_array,
    #         'pose_valid': dinov2_array,
    #         'lmk': dinov2_array,
    #         'lmk_valid': dinov2_array,
    #         'imagename': image_name,
    #         'dataset': image_name,
    #         'flame': flame,
    #         'exp': flame,
    #         'currpredmesh': dinov2_array,
    #         'bestallpredmesh': dinov2_array,
    #         'actorpredmesh': dinov2_array,
    #         'actorpredflame': dinov2_array,
    #         'gtmesh': dinov2_array,
    #         'gtflame': dinov2_array
    #     }