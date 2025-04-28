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

import os
import random
import sys
import math
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning import seed_everything

import src.datasets as datasets
from src.configs.config import cfg
from src.utils import util
import trimesh
import shutil

sys.path.append("./src")
from validator_pretained import Validator

torch.backends.cudnn.benchmark = False

def print_info(rank):
    props = torch.cuda.get_device_properties(rank)
    logger.info(f'[INFO]            {torch.cuda.get_device_name(rank)}')
    logger.info(f'[INFO] Rank:      {str(rank)}')
    logger.info(f'[INFO] Memory:    {round(props.total_memory / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Allocated: {round(torch.cuda.memory_allocated(rank) / 1024 ** 3, 1)} GB')
    logger.info(f'[INFO] Cached:    {round(torch.cuda.memory_reserved(rank) / 1024 ** 3, 1)} GB')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        cudnn.deterministic = True
        cudnn.benchmark = False

class TrainerFlame(object):
    def __init__(self, model, pretrainedmodel=None, config=None, device=None):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config

        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        self.seed = config.seed
        if config.withseed:
            seed_everything(self.seed)

        self.device = device
        self.batch_size = self.cfg.dataset.batch_size
        self.n_images = self.cfg.dataset.n_images
    
        print(self.cfg, flush=True)


        # autoencoder model
        self.model = model.to(self.device)
        
        # Print all modules in the model
        print("Model structure:")
        for name, module in self.model.named_children():
            print(f"- {name}: {type(module).__name__}")
            if hasattr(module, "named_children"):
                for sub_name, sub_module in module.named_children():
                    print(f"  - {sub_name}: {type(sub_module).__name__}")

        self.faces = model.flame.faces_tensor.cpu().numpy()
        self.load_checkpoint()

        print_info(device)
    
    def pretraining(self):
        self.validator = Validator(self)
        self.validator.prepare_data(with_exp=True)
        self.validator.run()

        
    # def load_checkpoint(self):
    #     self.epoch = 0
    #     self.global_step = 0
    #     dist.barrier()
    #     map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        
    #     # load pretrained model
    #     model_path = os.path.join(self.cfg.pretrained_model_path,'model_train_flameparamdiffusion_exp_53.tar')
    #     if model_path is not None and os.path.exists(model_path):
    #         logger.info(f'[TRAINER] Loading pretrained model from {model_path}')
    #         checkpoint = torch.load(model_path, map_location=map_location)
    #         print(checkpoint.keys(),'success checkpoint')


    def load_checkpoint(self):
        self.epoch = 0
        self.global_step = 0
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.device}
        
        # load pretrained model
        model_path = os.path.join(self.cfg.pretrained_model_path,'model_train_flameparamdiffusion_exp_53.tar')
        if model_path is not None and os.path.exists(model_path):
            logger.info(f'[TRAINER] Loading pretrained model from {model_path}')
            checkpoint = torch.load(model_path, map_location=map_location)
            print(checkpoint.keys(),'success checkpoint')
            
            # Load model components one by one
            try:
                # Load FARL model (named differently in your model)
                if 'farl' in checkpoint and hasattr(self.model, 'farlmodel'):
                    self.model.farlmodel.load_state_dict(checkpoint['farl'], strict=False)
                    logger.info("Loaded FARL weights")
                    
                # Load arcface
                if 'arcface' in checkpoint and hasattr(self.model, 'arcface'):
                    self.model.arcface.load_state_dict(checkpoint['arcface'], strict=False)
                    logger.info("Loaded ArcFace weights")
                    
                # Load network
                if 'net' in checkpoint and hasattr(self.model, 'net'):
                    self.model.net.load_state_dict(checkpoint['net'],strict=False)
                    logger.info("Loaded UNet weights")
                    
                # Load variance schedule
                if 'var_sched' in checkpoint and hasattr(self.model, 'var_sched'):
                    self.model.var_sched.load_state_dict(checkpoint['var_sched'],strict=False)
                    logger.info("Loaded variance schedule weights")
                    
                # Load diffusion model
                if 'diffusion' in checkpoint and hasattr(self.model, 'diffusion'):
                    self.model.diffusion.load_state_dict(checkpoint['diffusion'],strict=False)
                    logger.info("Loaded diffusion model weights")
                    
                logger.info("Successfully loaded all model components for inference")
                
                # Set model to evaluation mode for inference
                self.model.eval()
                
            except Exception as e:
                logger.error(f"Error loading pretrained model: {str(e)}")
                logger.error("Model loading failed, continuing with initialized weights")




