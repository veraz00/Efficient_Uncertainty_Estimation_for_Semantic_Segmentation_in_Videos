import os 
import torch 
import timeit 
import gc
import json 
import torch.nn.functional as F 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import ConfusionMatrix
import icecream
import _init_path
from .utils import * 

from datasets.utils import * 
from .optical_util import * 
import configs.constants as C  


def train(args, dataloaders, model, optimizer, criterion, \
          cur_loss, device, logger, output_dir):

    torch.cuda.empty_cache()
    gc.collect()
    start = timeit.default_timer()
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(args.training.start_epoch, args.training.epochs):
        logger.info(f'Start epoch {epoch}')
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                start_training_time = timeit.default_timer()
            else:
                model.eval()
            print(f'start {phase} phase')
            
            
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
                    scheduler.step(loss)

                    torch.save(model, os.path.join(output_dir, f'checkpoint.pt'))
                    checkpoint_tar_path = os.path.join(
                        output_dir, "checkpoint.pth.tar"
                    )
                    torch.save(
                        {
                            "loss": loss.item(),
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "last_epoch":epoch
                        },
                        checkpoint_tar_path,
                    )

                    if loss.item() < cur_loss:
                        torch.save(model, os.path.join(output_dir, "best.pt"))
                        cur_loss = loss
                        print(f"update the best.pt with {cur_loss}")

                        best_tar_path = os.path.join(
                            output_dir, "best.pth.tar"
                        )
                        torch.save(
                            {
                                "loss": cur_loss,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "last_epoch": epoch,
                            },
                            best_tar_path,
                        )


                if i > 0 and (i % args.training.log_batch_interval == 0 or i == len(dataloaders[phase])-1):
                    end_training_time = timeit.default_timer()
                    logger.info(f'Phase: {phase}\tEpoch: [{epoch}][{i}/{len(dataloaders[phase])-1}]\tLoss {loss.item()}\tTraining Time:{end_training_time-start_training_time}')
                    start_training_time = end_training_time
                    
    end = timeit.default_timer()
    logger.info(f'seconds to the whole train:{end -start} s')




def get_test(args, model, split, device, logger, sv_pred = True):


    frame_names = json.load(open(C.data_split_path))['val']['labeled']
    
    frame_names = list(frame_names)
    label_names = list(map(lambda x: x.replace(split, split + 'annot'), frame_names))
    model.eval()

    if args.model == 'bayesian_tiramisu':
        model.apply(set_dropout2d)
        logger.info(f'use dropout2d in {args.model}')
    
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
    
    threshold = args.error_thres 
    alpha_normal = args.alpha_normal 
    alpha_error = args.alpha_error 

    prev_img, prev_output_mean = None, None
    gts, preds, uncts = [], [], []
    uncts_r, uncts_e, uncts_v, uncts_b = [], [], [], []
    video_name = ''
    inference_time = 0
    cm = ConfusionMatrix(task="multiclass", num_classes=args.num_classes, \
        ignore_index = args.void_class).to(device)

    
    for i, frame_name in enumerate(frame_names): # frame_name: train/01TP_30_1860/01TP_000030.png
        torch.cuda.synchronize()
        t1 = time.time()
        img_path = os.path.join(args.dataset_dir, frame_name)
        label_path = os.path.join(args.dataset_dir , label_names[i])
        # img_gray = load_img(img_path, crop_size = (int(args.img_width), int(args.img_height)), \
        #                 return_gray = True)  # h, w
        img = load_img(img_path, crop_size = (int(args.img_width), int(args.img_height))) # h, w, c
        label = load_label(label_path, crop_size = (int(args.img_width), int(args.img_height))) # h, w, 

        img_tensor = image_process(img).to(device)  # b, c, w, h,
        label_tensor = label_process(label).to(device) # b, w, h


        # img_tensor_or = image_process(img, normalize = False) # b, c, w, h,

        T = args.sample_num
        for t in range(T):
            output = F.softmax(model(img_tensor), dim = 1)
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


        if args.use_warp: # the 2nd and other frames 

            

            if video_name != frame_name.split('/')[1]: # 01TP_30_1860 the 1st frame
                video_name = frame_name.split('/')[1]
                reconstruction_loss = None 

            if prev_img is not None:
                if args.flow == 'DF':
                    flow = cal_flow('DF', prev_img, img, \
                                    save_path = os.path.join(C.save_optical_flow, frame_name))  # h, w, 2
             
                warp_img = warp_frame(prev_img, flow, save_path = os.path.join(C.save_warped, frame_name)) # h, w, c
      
                reconstruction_loss = np.abs(warp_img - img).mean()
                mask = (reconstruction_loss < threshold) * 1
                alpha = mask * alpha_normal + (1-mask)*alpha_error


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


        if args.acqu_func != 'all':
            unc_map = acquisition_func(args.acqu_func, output_mean,\
                                   square_mean=square_mean, entropy_mean=entropy_mean) 
            unc_map = unc_map.squeeze().cpu().numpy()
            uncts.append(unc_map)
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
            uncts_r.append(unc_map_r)
            uncts_e.append(unc_map_e)
            uncts_b.append(unc_map_b)
            uncts_v.append(unc_map_v)


            
            # prev_frame_gray = img_gray


        # pred_np = np.asarray(np.argmax(torch2np(output_mean), axis=1), dtype=np.uint8).transpose(0, 2, 1)
        cm.update(output_mean, label_tensor)

        if sv_pred:
            sv_path = os.path.join(args.output_dir, 'testval_results')
            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            pred_index = np.argmax(torch2np(output_mean), axis=1).squeeze().T
            save_pred(pred_index, label, img, sv_path, frame_name, args.num_classes)


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
    if args.acqu_func != 'all':
        np.save(os.path.join(args.out_unct_dir, out_name), uncts)
    else:
        np.save(os.path.join(args.out_unct_dir_r, out_name), uncts_r)
        np.save(os.path.join(args.out_unct_dir_e, out_name), uncts_e)
        np.save(os.path.join(args.out_unct_dir_b, out_name), uncts_b)
        np.save(os.path.join(args.out_unct_dir_v, out_name), uncts_v)

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