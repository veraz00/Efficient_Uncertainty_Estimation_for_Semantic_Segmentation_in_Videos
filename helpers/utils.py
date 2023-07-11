import logging 
import os 
import time 
import icecream
import numpy as np
import cv2 
import torch.nn as nn 
import torch 
from torchmetrics import ConfusionMatrix
import configs.constants as C 

def create_logger(cfg, phase = 'train'):
    print('create logger')

    log_dir = os.path.join(cfg.output_dir, cfg.dataset.name, cfg.model.name)
    os.makedirs(log_dir, exist_ok = True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(phase, time_str)
    final_log_file = os.path.join(log_dir, log_file)
    head  = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename = final_log_file, format = head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger, log_dir


def init_seed(seed = 0):
    print('init seed')
    import random 
    import numpy as np 
    import torch 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return 


def set_dropout2d(m):
    if type(m) == nn.Dropout2d:
        m.train
    

def torch2np(val):
    try:
        return val.item().detach().cpu()
    except:
        return val.detach().numpy()

def np2tensor(val, device = None):
    if device == None:
        return torch.from_numpy(val)
    else:
        return torch.from_numpy(val).to(device)




def id_to_rgb(label, n_classes):
    rgb_label = np.zeros((label.shape[0], label.shape[1], 3))
    for i in range(n_classes):
        rgb_label[label == i] = C.index_to_color[i]
    rgb_label = rgb_label.astype('uint8')

    return rgb_label



def save_pred(pred, label, image, sv_path, img_name, n_classes):
    """
    pred: 224, 224 # h, w
    label: 224, 224 
    image: 224, 224, 3 
    """
    try:
        # save rgb
        img_path = os.path.join(sv_path, 'rgb', img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, image)

        # save pred label_data
        # color_map = C.get_id_color_map()
        pred_color = id_to_rgb(pred, n_classes)
        pred_path = os.path.join(sv_path, 'pred', img_name)
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        cv2.imwrite(pred_path, pred_color)


        gt_color = id_to_rgb(label, n_classes)
        gt_path = os.path.join(sv_path, 'gt', img_name)
        os.makedirs(os.path.dirname(gt_path), exist_ok=True)
        cv2.imwrite(gt_path, gt_color)

    except Exception as e:
        print("Exception in save_pred: ", e)



def image_process(img, normalize=True):
    img = img.astype(np.float64)
    if normalize:
        img = img.astype(float) / 255.0
        img -= C.mean
        img /= C.std
    # NHWC -> NCHW
    img_torch = img.transpose(2, 1, 0)
    img_torch = np.expand_dims(img_torch, 0)
    img_torch = torch.from_numpy(img_torch).float()

    return img_torch

def label_process(label,):
    label_torch = label.T 
    label_torch = np.expand_dims(label_torch, axis=(0, ))
    label_torch = torch.from_numpy(label_torch).float()
    return label_torch


def calculation_from_confusion_matrix(
    cm, ignore_index = 0, return_iou_only=False
):
    #  ___ pre
    # |
    # |
    # gt

    cm_t = cm.compute()
    if ignore_index == 0:
        cm_t = cm_t[1:, 1:]
    
    res = torch.sum(cm_t, 0)  # for every class on gt
    pos = torch.sum(cm_t, 1)  # for every class on pre
    TP = torch.diagonal(cm_t)
    FN = torch.sum(cm_t, 1) - TP
    FP = torch.sum(cm_t, 0) - TP
    TN = torch.sum(cm_t) - (FP + FN + TP)

    # ACC
    PIXEL_ACC = torch.sum(TP) / torch.sum(pos)
    
    AVG_ACC = torch.nanmean(torch.where(pos>0, TP/pos, torch.nan))  # TP/np.maximum(1.0, pos) same as TPR
    # all = torch.where(pos + res - TP > 0, pos + res - TP, 1.0)
    # IOU_ARRAY = TP / all  # TPR
    all = pos + res - TP 
    IOU_ARRAY = torch.where(all > 0, TP/all, torch.nan)

    AVG_IOU = torch.nanmean(IOU_ARRAY)

    if return_iou_only:
        return AVG_IOU, IOU_ARRAY
    # Sensitivity, hit rate, recall, or true positive rate  # what we care for auto driving
    TPR = torch.where((TP + FN) > 0, TP/(TP + FN), torch.nan)
    # Specificity or true negative rate
    TNR = torch.where((TN + FP) > 0, TN/(TP + FN), torch.nan)
    # Precision or positive predictive value
    PPV = torch.where((TP + FP) > 0, TP/(TP + FN), torch.nan)
    # Negative predictive value
    NPV = torch.where((TN + FN) > 0, TN/(TN + FN), torch.nan)
    # Fall out or false positive rate
    FPR = torch.where((FP + TN) > 0, FP/(TN + FP), torch.nan)
    # False negative rate
    FNR = torch.where((TP + FN) > 0, FN/(TP + FN), torch.nan)
    # False discovery rate
    FDR = torch.where((TP + FP) > 0, FP/(TP + FP), torch.nan)

    # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    # if have_ignore_index:
    #     class_len = len(TPR) - 1
    # else:
    class_len = len(TPR)

    AVG_TPR = torch.nanmean(TPR) 
    AVG_TNR = torch.nanmean(TNR) 
    AVG_PPV = torch.nanmean(PPV) 
    AVG_NPV = torch.nanmean(NPV) 
    AVG_FPR = torch.nanmean(FPR) 
    AVG_FNR = torch.nanmean(FNR) 
    AVG_FDR = torch.nanmean(FDR) 

    metrices = {
        "TPR": TPR,
        "TNR": TNR,
        "PPV": PPV,
        "NPV": NPV,
        "FPR": FPR,
        "FNR": FNR,
        "FDR": FDR,
        "PIXEL_ACC": PIXEL_ACC,
        "IOU_ARRAY": IOU_ARRAY,
        "AVG_TPR": AVG_TPR,
        "AVG_TNR": AVG_TNR,
        "AVG_PPV": AVG_PPV,
        "AVG_NPV": AVG_NPV,
        "AVG_FPR": AVG_FPR,
        "AVG_FNR": AVG_FNR,
        "AVG_FDR": AVG_FDR,
        "AVG_ACC": AVG_ACC,
        "AVG_IOU": AVG_IOU,
    }
    return metrices, cm_t