import argparse
import torch 
from torch.utils.data import DataLoader
import torch.nn as nn 
import pprint
import os

import _init_path

from configs import cfg, update_config  # cfg 
from utils.utils import * 
from utils.function import train 
from models import get_model
from datasets.camvid_datasets import Camvid

def parse_args():
    parser = argparse.ArgumentParser(description='Python3 for uncertainty estimation')
    parser.add_argument('--cfg_file_path', default = '', type = str, help = 'path to config file')
    parser.add_argument('opts', help = 'modify the config', nargs = argparse.REMAINDER)
    args = parser.parse_args()

    if args.cfg_file_path != '':
        update_config(cfg, args)
    return args 




if __name__ == '__main__':

    args = parse_args()
    logger, output_dir = create_logger(cfg, phase = 'train')
    logger.info(pprint.pformat(args))
    logger.info(cfg)
    init_seed(cfg.seed)

    train_dataset = Camvid(root = cfg.dataset_dir,
                          split = 'train',
                          is_aug = True,
                          img_size = (cfg.img_width, cfg.img_height)
                        ) # get_dataset(cfg.dataset_name)
    logger.info('train dataset: {} num_images: {}'.format(cfg.dataset_name, len(train_dataset)))
    logger.info('class weughts: {}'.format(train_dataset.class_weights))
    
    val_dataset = Camvid(root = cfg.dataset_dir,
                            split = 'val',
                            is_aug = False,
                            img_size = (cfg.img_width, cfg.img_height),
                            class_weights = train_dataset.class_weights
                            ) # get_dataset(cfg.dataset_name)



    dataloaders = {
                    'train': DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle = True, \
                                             num_workers = cfg.num_workers, pin_memory = True), 
                    'valid': DataLoader(val_dataset, batch_size = cfg.batch_size, shuffle = False, \
                                             num_workers = cfg.num_workers, pin_memory = True)
                    }
    


    if cfg.model == 'bayesian_tiramisu':
        model = get_model(cfg.model, cfg.n_classes)


    device = torch.device('cuda' if torch.cuda.is_available() and cfg.device != 'cpu' else 'cpu')
    
    if cfg.device != 'cpu' and len(cfg.gpus) > 1:
        model = nn.DataParallel(model, device_ids = cfg.gpus)
    model = model.to(device)

    cfg.start_epoch = 0 
    if cfg.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr = cfg.lr, momentum = 0.9, weight_decay = cfg.weight_decay)
    elif cfg.opt == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr = cfg.lr, momentum = 0.9, weight_decay = cfg.weight_decay)

    cur_loss = torch.float('inf')
    if cfg.resume:
        if os.path.exists(cfg.resume_path) == False:
            raise ValueError('resume_path should not be empty')
        logger.info('resuming finetune from {}'.format(cfg.resume_path))
        checkpoint = torch.load(cfg.resume_path)
        if checkpoint.endswith('.pth'):
            cfg.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            cur_loss = checkpoint['loss']

        else:
            model = torch.load(checkpoint)

        logger.info('resuming finetune from {} done'.format(cfg.resume_path))
        logger.info('start_epoch: {}'.format(cfg.start_epoch))
    
    criterion = nn.CrossEntropyLoss(weight = torch.tensor(train_dataset.class_weights).to(device))
    train(cfg, dataloaders, model, optimizer, criterion, \
          cur_loss, device, logger, output_dir)
    logger.info('Finished training')









