from yacs.config import CfgNode as CN 

cfg = CN()

cfg.seed = 705
cfg.device = 'gpu'
cfg.gpus = (0,)
cfg.output_dir = '/home/linlin/workspace/Efficient_p3/outputs'
cfg.num_workers = 8

# cudnn 
cfg.cudnn = CN()
cfg.cudnn.benchmark = True
cfg.cudnn.deterministic = False
cfg.cudnn.enabled = True

# dataset
cfg.dataset = CN()
cfg.dataset.name = 'camvid'
cfg.dataset.dataset_dir = '/home/linlin/dataset/camvid_video'
cfg.dataset.img_height = 128
cfg.dataset.img_width = 64
cfg.dataset.num_classes = 12
cfg.dataset.void_class = 0




# model
cfg.model = CN()
cfg.model.type = 'bayesian_tiramisu'
cfg.model.name = 'bayesian_tiramisu_lr0.01_b4'
cfg.model.is_bayesian = True



# training
cfg.training = CN()
cfg.training.resume = False
cfg.training.resume_path = '/home/linlin/workspace/Efficient_p3/outputs/camvid/bayesian_tiramisu_lr0.01_b4/checkpoint.pt'
cfg.training.epochs = 100
cfg.training.lr = 0.005
cfg.training.weight_decay = 1e-4
cfg.training.batch_size = 2
cfg.training.opt = 'rms'
cfg.training.log_batch_interval = 100
cfg.training.save_epoch_interval = 1


cfg.validation = CN()
cfg.validation.save_epoch_interval = 50

# testing
cfg.test = CN()
cfg.test.test_model_path = '/home/linlin/workspace/Efficient_p3/outputs/camvid/bayesian_tiramisu_lr0.001_b4/checkpoint.pth.tar'
cfg.test.use_warp = True
cfg.test.flow = 'DF' # 'flownet2' 
cfg.test.flow_model_path = ''
cfg.test.sample_num= 4
cfg.test.test_video = False
cfg.test.acqu_func = 'all'
cfg.test.error_thres = 40
cfg.test.alpha_normal = 0.2
cfg.test.alpha_error = 0.5
cfg.test.save_output = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg_file_path)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

if __name__ == '__main__':
    print(cfg.dump())  # print formatted configs
    with open("/home/linlin/workspace/Efficient_p3/configs/default.yaml", "w") as f:
        f.write(cfg.dump())   # save config to file