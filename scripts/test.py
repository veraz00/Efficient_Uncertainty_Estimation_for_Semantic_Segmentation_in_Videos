

import os 
import torch 
import torch.nn as nn 
import argparse 
import icecream as ic 

import _init_path
from helpers.function import get_test 
from configs import cfg, update_config 

from helpers.utils import * 
from datasets.get_dataset import dataset_dict
from models import * 

def parse_args():
    parser = argparse.ArgumentParser(description='Python3 Test for uncertainty estimation ')
    parser.add_argument('--cfg_file_path', default = '/home/linlin/workspace/Efficient_p3/configs/default.yaml', type = str, help = 'path to config file')
    parser.add_argument('--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--use_warp', type = bool, default = False, help = 'use warp or not')
    parser.add_argument('opts', help = 'modify the config', nargs = argparse.REMAINDER)
    args = parser.parse_args()
    update_config(cfg, args)
    return args 


if __name__ == '__main__':
    
    args = parse_args()

    logger, output_dir = create_logger(cfg, phase = 'test')
    init_seed(cfg.seed)
    logger.info(cfg)


    test_dataset = dataset_dict(cfg.dataset.name)


    dataset = test_dataset(root = cfg.dataset.dataset_dir,
                          split = 'val',
                          img_size = (cfg.dataset.img_width, cfg.dataset.img_height),
    )
    logger.info('Test dataset: {} num_images: {}'.format(cfg.dataset.name, len(dataset)))


    if cfg.model.type == 'bayesian_tiramisu':
        model = get_model(cfg.model.type, cfg.dataset.num_classes)



    device = torch.device('cuda' if torch.cuda.is_available() and cfg.device == 'gpu' else 'cpu')
    if device != 'cpu' and len(cfg.gpus) > 1:
        model = nn.DataParallel(model, device_ids = cfg.gpus)


    if os.path.exists(cfg.test.test_model_path) == False:

        raise ValueError('test_model_path should not be empty')
    else:
        logger.info('Test from {}'.format(cfg.test.test_model_path))
        checkpoint = torch.load(cfg.test.test_model_path, map_location=device)
        if cfg.test.test_model_path.endswith('.pth.tar'):
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model = torch.load(checkpoint)
    
    model.to(device)
    icecream.ic(device)
    
    logger.info('load test model from {} done'.format(cfg.test.test_model_path))

    # run validate
    mean_IoU, IoU_array, pixel_acc, mean_acc = get_test(cfg, model, split = 'val', device = device, logger = logger, )

    ic(mean_IoU, IoU_array, pixel_acc, mean_acc)
    logger.info('finish testing!')
