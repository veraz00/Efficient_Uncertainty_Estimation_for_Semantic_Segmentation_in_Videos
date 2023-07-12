import os 
import torch 
import timeit 
import gc
import json 
import torch.nn.functional as F 
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torchmetrics import ConfusionMatrix
import icecream
import _init_path
from .utils import * 

from datasets.utils import * 
from .optical_util import * 
import configs.constants as C  


def train(args, dataloaders, model, optimizer, criterion, \
          cur_miou, start_epoch, device, logger, output_dir):

    torch.cuda.empty_cache()
    gc.collect()
    start = timeit.default_timer()

    # optimizer.param_groups[0]["lr"] = args.training.lr
    # if len(optimizer.param_groups) == 2:
    #     optimizer.param_groups[1]["lr"] = args.training.lr
    # scheduler = ReduceLROnPlateau(optimizer, 'max')
    scheduler = StepLR(optimizer, step_size = 30, gamma = 0.5)

    for epoch in range(start_epoch, args.training.epochs):
        logger.info(f'Start epoch {epoch}')
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                
            else:
                model.eval()
                cm = ConfusionMatrix(task="multiclass", num_classes=args.dataset.num_classes, \
                    ignore_index = args.dataset.void_class).to(device)

            print(f'start {phase} phase')
            
            start_training_time = timeit.default_timer()
            for i, (images, labels, img_path) in enumerate(dataloaders[phase]):
                images = images.to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                if phase == 'valid' and epoch % args.validation.save_epoch_interval == 0:
                    # scheduler.step(loss)
                    cm.update(outputs, labels)

                if (phase == 'train' and i > 0 and (i % args.training.log_batch_interval == 0)) or \
                    (phase == 'valid' and i > 0 and (i % args.validation.log_epoch_interval == 0)) or \
                    (i == len(dataloaders[phase])-1):
                    end_training_time = timeit.default_timer()
                    lr = optimizer.param_groups[-1]['lr']

                    logger.info(f'Phase: {phase}\tEpoch: [{epoch}]\tBatch:[{i}/{len(dataloaders[phase])-1}]\t\
                                lr:{lr}\tLoss {loss.item()}\t\
                                Time:{end_training_time-start_training_time}')
                    start_training_time = end_training_time
                    
        if phase == 'valid':
            scheduler.step()  # pass it by step_size times, lr *= gamma

        if epoch % args.validation.save_epoch_interval == 0:
            metrics_valid, cm_valid = calculation_from_confusion_matrix(cm)
            logger.info(cm_valid)
            logger.info(metrics_valid)
        
            torch.save(model, os.path.join(output_dir, f'checkpoint.pt'))
            checkpoint_tar_path = os.path.join(
                output_dir, "checkpoint.pth.tar"
            )

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "last_epoch":epoch,
                    "miou": cur_miou,
                    # 'lr':optimizer.param_groups[-1]['lr']
                },
                checkpoint_tar_path,
            )
            logger.info(f'update {checkpoint_tar_path} on epoch{epoch}')

            if metrics_valid['AVG_IOU'] > cur_miou:
                torch.save(model, os.path.join(output_dir, "best.pt"))
                cur_miou = metrics_valid['AVG_IOU'] 


                best_tar_path = os.path.join(
                    output_dir, "best.pth.tar"
                )
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "last_epoch": epoch,
                        "miou": cur_miou,
                        # 'lr':optimizer.param_groups[-1]['lr']
                    },
                    best_tar_path,
                )
                logger.info(f'update {best_tar_path} on epoch{epoch} with miou={cur_miou}')

                    

    end = timeit.default_timer()
    logger.info(f'seconds to the whole train:{end -start} s')




def get_test(args, model, split, device, logger, sv_pred = True):


    frame_names = json.load(open(C.data_split_path))['val']['labeled']
    
    frame_names = list(frame_names)[:5]
    label_names = list(map(lambda x: x.replace(split, split + 'annot'), frame_names))
    model.eval()

    if args.model.type == 'bayesian_tiramisu':
        model.apply(set_dropout2d)
        logger.info(f'use dropout2d in {args.model.type}')
    
    # if args.flow == 'DF':
    #     DF = cv2.optflow.createOptFlow_DeepFlow()
    # if args.float == 'flownet2':

    #     flownet2 = FlowNet2()
    #     flownet2_path = args.flow_model_path 
    #     pretrained_dict = torch.load(flownet2_path)['state_dict']
    #     model_dict = flownet2.state_dict()
    #     pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     flownet2.load_state_dict(model_dict)
    #     flownet2 = flownet2.to(device)
    
    threshold = args.test.error_thres 
    alpha_normal = args.test.alpha_normal 
    alpha_error = args.test.alpha_error 

    prev_img, prev_output_mean = None, None
    gts, preds, uncts = [], [], []
    uncts_r, uncts_e, uncts_v, uncts_b = [], [], [], []
    video_name = ''
    inference_time = 0
    cm = ConfusionMatrix(task="multiclass", num_classes=args.dataset.num_classes, \
        ignore_index = args.dataset.void_class).to(device)

    torch.cuda.empty_cache()
    for i, frame_name in enumerate(frame_names): # frame_name: train/01TP_30_1860/01TP_000030.png
        torch.cuda.synchronize()
        t1 = time.time()
        img_path = os.path.join(args.dataset.dataset_dir, frame_name)
        label_path = os.path.join(args.dataset.dataset_dir , label_names[i])
        # img_gray = load_img(img_path, crop_size = (int(args.img_width), int(args.img_height)), \
        #                 return_gray = True)  # h, w
        img = load_img(img_path, crop_size = (int(args.dataset.img_width), int(args.dataset.img_height))) # h, w, c
        label = load_label(label_path, crop_size = (int(args.dataset.img_width), int(args.dataset.img_height))) # h, w, 

        img_tensor = image_process(img).to(device)  # b, c, w, h,
        label_tensor = label_process(label).to(device) # b, w, h


        # img_tensor_or = image_process(img, normalize = False) # b, c, w, h,

        T = args.test.sample_num
        for t in range(1):
            output = F.softmax(model(img_tensor), dim = 1)
            print('get output')
            if t == 0:
                output_mean = output * 0
                output_square = output * 0
                entropy_mean = output.mean(1) * 0
            output_mean += output
            output_square += output.pow(2)
            entropy_mean += acquisition_func('e', output)
        output_mean = output_mean / T  # (1, 12, 224, 224)
        square_mean = output_square / T

        entropy_mean = entropy_mean / T # (1, 224, 224)


        if args.test.use_warp: # the 2nd and other frames 
            logger.info('use warp to test')

            

            if video_name != frame_name.split('/')[1]: # 01TP_30_1860 the 1st frame
                video_name = frame_name.split('/')[1]
                reconstruction_loss = None 

            if prev_img is not None:
                if args.test.flow == 'DF':
                    flow = cal_flow('DF', prev_img, img, \
                                    save_path = os.path.join(C.save_optical_flow, frame_name))  # h, w, 2
             
                warp_img = warp_frame(prev_img, flow, save_path = os.path.join(C.save_warped, frame_name)) # h, w, c
      
                reconstruction_loss = np.abs(warp_img - img).mean(axis = -1).T
                # reconstruction_loss = np.expand_dims(reconstruction_loss, axis = (0, 1)) # 64, 128, 
                # reconstruction_loss = np2tensor(reconstruction_loss, device)
                icecream.ic(np.count_nonzero(reconstruction_loss < threshold) * 1)
                icecream.ic(np.count_nonzero(reconstruction_loss < threshold) * 1)
                mask = np2tensor((reconstruction_loss < threshold) * 1, device=device)

                alpha = mask * alpha_normal + (1-mask)*alpha_error
                icecream.ic(alpha.shape, output_mean.shape)


                output_np_warped = warp_prediction(prev_output_mean.transpose(-1, -2), np2tensor(flow, device = device).unsqueeze(0))  
                # n, c, h, w, 
                # flow: n, h, w, 2 
                output_mean_warped = output_np_warped.transpose(-1, -2)
                icecream.ic(output_mean_warped.shape)


                output_mean = output_mean_warped * (1 - alpha) + output_mean * alpha
                square_mean = torch.pow(output_mean_warped, 2) * (1 - alpha) + square_mean * alpha
                icecream.ic(acquisition_func('e', output_mean_warped).shape)
                entropy_mean = acquisition_func('e', output_mean_warped)* (1 - alpha) + entropy_mean * alpha



            prev_img = img
            prev_output_mean = output_mean


        if args.test.acqu_func != 'all':
            unc_map = acquisition_func(args.test.acqu_func, output_mean,\
                                   square_mean=square_mean, entropy_mean=entropy_mean) 
            unc_map = unc_map.squeeze().cpu().numpy()
            uncts.append(torch2np(unc_map))
        else:
            unc_map_r = acquisition_func('r', output_mean,\
                                    square_mean=square_mean, entropy_mean=entropy_mean)
            unc_map_e = acquisition_func('e', output_mean,\
                                    square_mean=square_mean, entropy_mean=entropy_mean)
            unc_map_b = acquisition_func('b', output_mean,\
                                    square_mean=square_mean, entropy_mean=entropy_mean)
            unc_map_v = acquisition_func('v', output_mean,\
                                    square_mean=square_mean, entropy_mean=entropy_mean)
            # unc_map_r = unc_map_r.squeeze().cpu().numpy()
            # unc_map_e = unc_map_e.squeeze().cpu().numpy()
            # unc_map_b = unc_map_b.squeeze().cpu().numpy()
            # unc_map_v = unc_map_v.squeeze().cpu().numpy()
            uncts_r.append(torch2np(unc_map_r))
            uncts_e.append(torch2np(unc_map_e))
            uncts_b.append(torch2np(unc_map_b))
            uncts_v.append(torch2np(unc_map_v))


            
        #     # prev_frame_gray = img_gray


        # pred_np = np.asarray(np.argmax(torch2np(output_mean), axis=1), dtype=np.uint8).transpose(0, 2, 1)
        cm.update(output_mean, label_tensor)

        if sv_pred:
            sv_path = os.path.join(args.output_dir, 'testval_results')
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            pred_index = np.argmax(torch2np(output_mean), axis=1).squeeze().T
            save_pred(pred_index, label, img, sv_path, frame_name, args.dataset.num_classes)


        if i % 100 == 0:

            logging.info('processing: %d images' % i)
            AVG_IOU, IOU_ARRAY = calculation_from_confusion_matrix(cm, return_iou_only= True)
            logging.info(f'{i}th img: {frame_name}, mIoU: {AVG_IOU}, iou_array: {IOU_ARRAY}')

    torch.cuda.synchronize()
    t2 = time.time()
    inference_time += t2 - t1   

    logger.info(f"     + Done {len(frame_names)} iterations inference !")
    logger.info("     + Total time cost: {}s".format(t2-t1))
    logger.info("     + Average time cost: {}s".format((t2-t1) / len(frame_names)))
    logger.info("     + Frame Per Second: {:.2f}".format(1 / ((t2-t1)/ len(frame_names))))




    out_name = 'uncts'
    if args.test.acqu_func != 'all':
        os.makedirs(args.test.out_unct_dir, exist_ok= True)
        np.save(os.path.join(args.test.out_unct_dir, out_name), uncts)
    else:
        os.makedirs(args.test.out_unct_dir_r, exist_ok= True)
        os.makedirs(args.test.out_unct_dir_e, exist_ok= True)
        os.makedirs(args.test.out_unct_dir_b, exist_ok= True)
        os.makedirs(args.test.out_unct_dir_v, exist_ok= True)

        np.save(os.path.join(args.test.out_unct_dir_r, out_name), uncts_r)
        np.save(os.path.join(args.test.out_unct_dir_e, out_name), uncts_e)
        np.save(os.path.join(args.test.out_unct_dir_b, out_name), uncts_b)
        np.save(os.path.join(args.test.out_unct_dir_v, out_name), uncts_v)

    metrices, cm_t = calculation_from_confusion_matrix(cm)
    mean_IoU = metrices['AVG_IOU']
    IoU_array = metrices['IOU_ARRAY']
    pixel_acc = metrices['PIXEL_ACC']
    mean_acc = metrices['AVG_ACC']
    logging.info('CM\n')
    logging.info(cm_t)
    logging.info(metrices)

    logging.info(f'mean_IoU: {mean_IoU}\nIoU_array: {IoU_array}\npixel_acc:{pixel_acc}\nmean_acc:{mean_acc}')
    return mean_IoU, IoU_array, pixel_acc, mean_acc