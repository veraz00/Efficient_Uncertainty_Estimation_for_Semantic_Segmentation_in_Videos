from yacs.config import CfgNode as CN 

cfg = CN()

cfg.seed = 705
# basic 
cfg.use_warp = True 

# dataset
cfg.dataset_name = 'camvid'
cfg.dataset_dir = '/home/linlin/dataset/camvid_video'
cfg.img_height = 128
cfg.img_width = 64
cfg.num_classes = 12
cfg.void_class = 0
cfg.num_workers = 8



# model
cfg.model = 'bayesian_tiramisu'
cfg.model_name = 'bayesian_tiramisu_lr0.001_b4'
cfg.is_bayesian = True
cfg.resume = False
cfg.resume_path = '/home/linlin/workspace/Efficient_p3/outputs/checkpoint.pth.tar'
cfg.device = 'cpu'
cfg.gpus = (0,)
cfg.epochs = 100
cfg.lr = 0.001
cfg.weight_decay = 1e-4
cfg.batch_size = 2
cfg.opt = 'rms'

cfg.log_batch_interval = 100
cfg.save_epoch_interval = 1
cfg.output_dir = '/home/linlin/workspace/Efficient_p3/outputs'


# testing
cfg.use_warp = False
cfg.flow = 'DF' # 'flownet2' # 

cfg.sample_num= 4
cfg.test_video = False
cfg.test_model_path = '/home/linlin/workspace/Efficient_p3/outputs/camvid/bayesian_tiramisu_lr0.001_b4/checkpoint.pth.tar'
cfg.flow_model_path = '/home/linlin/workspace/Efficient_p3/pretrained/FlowNet2_checkpoint.pth.tar'
cfg.acqu_func = 'all'
cfg.error_thres = 40
cfg.alpha_normal = 0.2
cfg.alpha_error = 0.5
cfg.save_output = False


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg_file_path)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg