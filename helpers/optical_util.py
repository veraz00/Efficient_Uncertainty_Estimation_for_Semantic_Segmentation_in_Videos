import numpy as np 
import torch
import torch.nn as nn
from torch.autograd import Variable



def image_process(img, normalize=True):

    mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
    std = [0.27413549931506, 0.28506257482912, 0.28284674400252]


    img = img.astype(np.float64)
    if normalize:
        img = img.astype(float) / 255.0
        img -= mean
        img /= std
    # NHWC -> NCHW
    img_torch = img.transpose(2, 1, 0)
    img_torch = np.expand_dims(img_torch, 0)
    img_torch = torch.from_numpy(img_torch).float()

    return img_torch

def acquisition_func(acqu, output_mean, square_mean=None, entropy_mean=None):
    if acqu == 'e':  # max entropy
        return -(output_mean * torch.log(output_mean)).mean(1)
    elif acqu == 'b':  
        return acquisition_func('e', output_mean) - entropy_mean
    elif acqu == 'r':  # variation ratios
        return 1 - output_mean.max(1)[0]
    elif acqu == 'v':  # mean STD
        return (square_mean - torch.pow(output_mean, 2)).mean(1)


# def generate_meshgrid(flow):
#     h = flow.size(2)
#     w = flow.size(3)
#     y = torch.arange(0, h).unsqueeze(1).repeat(1, w) / (h - 1) * 2 - 1
#     x = torch.arange(0, w).unsqueeze(0).repeat(h, 1) / (w - 1) * 2 - 1
#     mesh_grid = Variable(torch.stack([x,y], 0).unsqueeze(0).repeat(flow.size(0), 1, 1, 1).cuda(), volatile=True)
#     return mesh_grid


# def warp_tensor(tensor, flow):
#     mesh_grid = generate_meshgrid(flow)
#     flow = flow.clone()
#     flow[:,0,:,:] = flow[:,0,:,:] / (tensor.size()[3] / 2)
#     flow[:,1,:,:] = flow[:,1,:,:] / (tensor.size()[2] / 2)
#     grid = flow + mesh_grid
#     grid = torch.transpose(torch.transpose(grid, 1, 2), 2, 3)
#     warped_tensor = nn.functional.grid_sample(tensor, grid)

#     return warped_tensor

import os
import cv2
import numpy as np
import flow_vis
import icecream

def cal_flow(method, prev, cur, save_path):
    """
    prev: grey, batch, h, w, c 
    """
    icecream.ic(prev.shape)
    icecream.ic(cur.shape)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    if method == 'DF':
        dis = cv2.optflow.createOptFlow_DeepFlow()
        flow = dis.calc(prev_gray, cur_gray, None)
        icecream.ic(flow.shape)

        flow_rgb = flow_vis.flow_to_color(flow, convert_to_bgr=False)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, flow_rgb)
    return flow 

def warp_frame(prev, flow, save_path = None):
    h, w = flow.shape[:2]
    # flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    cur = cv2.remap(prev, flow, None, cv2.INTER_LINEAR)
    if save_path != None:
        os.makedirs(os.path.dirname(save_path), exist_ok= True)
        cv2.imwrite(save_path, cur)
    icecream.ic(cur.shape)
    return cur 

from .utils import * 
def generate_meshgrid(flow):
    h, w = flow.shape[1], flow.shape[2]
    y = torch.arange(0, h).unsqueeze(1).repeat(1, w).float() / (h - 1) * 2 - 1
    x = torch.arange(0, w).unsqueeze(0).repeat(h, 1).float() / (w - 1) * 2 - 1 # 1, 128, 64  = n, h, w

    mesh_grid = torch.stack([x, y], dim=2).unsqueeze(0).repeat(flow.size(0), 1, 1, 1).to(device=flow.device) # n, h, w, 2 

    return mesh_grid


def warp_prediction(prev_prediction, flow):
    """
    prev_prediction: n, 12, h, w
    flow: n, h, w, 2

    output: n, 12, h, w
    """

    icecream.ic(prev_prediction.shape)
    icecream.ic(flow.shape)
    h, w = flow.shape[:2]
    grid = generate_meshgrid(flow)  # n, h, w, 2
    flow = flow.clone()
    flow[..., 0] = flow[..., 0] / (w / 2)
    flow[..., 1] = flow[..., 1] / (h / 2)
    grid += flow
    icecream.ic(grid.shape, prev_prediction.shape)


    # grid = grid.permute()
    warped_prediction = nn.functional.grid_sample(prev_prediction, grid, align_corners=True)


    return warped_prediction




# def flow2img(flow, BGR=True):
# 	x, y = flow[..., 0], flow[..., 1]
# 	hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
# 	ma, an = cv2.cartToPolar(x, y, angleInDegrees=True)
# 	hsv[..., 0] = (an / 2).astype(np.uint8)
# 	hsv[..., 1] = (cv2.normalize(ma, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)).astype(np.uint8)
# 	hsv[..., 2] = 255
# 	if BGR:
# 		img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# 	else:
# 		img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
# 	return img


def image_warp(im, flow, output_path, mode='bilinear'):
    """Performs a backward warp of an image using the predicted flow.
    numpy version

    Args:
        im: input image. ndim=2, 3 or 4, [[num_batch], height, width, [channels]]. num_batch and channels are optional, default is 1.
        flow: flow vectors. ndim=3 or 4, [[num_batch], height, width, 2]. num_batch is optional
        mode: interpolation mode. 'nearest' or 'bilinear'
    Returns:
        warped: transformed image of the same shape as the input image.
    """


            
    # assert im.ndim == flow.ndim, 'The dimension of im and flow must be equal '
    flag = 4
    if im.ndim == 2:
        height, width = im.shape
        num_batch = 1
        channels = 1
        im = im[np.newaxis, :, :, np.newaxis]
        flow = flow[np.newaxis, :, :]
        flag = 2
    elif im.ndim == 3:
        height, width, channels = im.shape
        num_batch = 1
        im = im[np.newaxis, :, :]
        flow = flow[np.newaxis, :, :]
        flag = 3
    elif im.ndim == 4:
        num_batch, height, width, channels = im.shape
        flag = 4
    else:
        raise AttributeError('The dimension of im must be 2, 3 or 4')

    max_x = width - 1
    max_y = height - 1
    zero = 0

    # We have to flatten our tensors to vectorize the interpolation
    im_flat = np.reshape(im, [-1, channels])
    flow_flat = np.reshape(flow, [-1, 2])

    # Floor the flow, as the final indices are integers
    flow_floor = np.floor(flow_flat).astype(np.int32)

    # Construct base indices which are displaced with the flow
    pos_x = np.tile(np.arange(width), [height * num_batch])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])
    pos_y = np.tile(np.reshape(grid_y, [-1]), [num_batch])

    x = flow_floor[:, 0]
    y = flow_floor[:, 1]
    print(flow_flat.shape)
    x0 = pos_x + x
    y0 = pos_y + y

    x0 = np.clip(x0, zero, max_x)
    y0 = np.clip(y0, zero, max_y)

    dim1 = width * height
    batch_offsets = np.arange(num_batch) * dim1
    base_grid = np.tile(np.expand_dims(batch_offsets, 1), [1, dim1])
    base = np.reshape(base_grid, [-1])

    base_y0 = base + y0 * width

    if mode == 'nearest':
        idx_a = base_y0 + x0
        warped_flat = im_flat[idx_a]
    elif mode == 'bilinear':
        # The fractional part is used to control the bilinear interpolation.
        bilinear_weights = flow_flat - np.floor(flow_flat)

        xw = bilinear_weights[:, 0]
        yw = bilinear_weights[:, 1]

        # Compute interpolation weights for 4 adjacent pixels
        # expand to num_batch * height * width x 1 for broadcasting in add_n below
        wa = np.expand_dims((1 - xw) * (1 - yw), 1) # top left pixel
        wb = np.expand_dims((1 - xw) * yw, 1) # bottom left pixel
        wc = np.expand_dims(xw * (1 - yw), 1) # top right pixel
        wd = np.expand_dims(xw * yw, 1) # bottom right pixel

        x1 = x0 + 1
        y1 = y0 + 1

        x1 = np.clip(x1, zero, max_x)
        y1 = np.clip(y1, zero, max_y)

        base_y1 = base + y1 * width
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

        warped_flat = wa * Ia + wb * Ib + wc * Ic + wd * Id
    warped = np.reshape(warped_flat, [num_batch, height, width, channels])

    if flag == 2:
        warped = np.squeeze(warped)
    elif flag == 3:
        warped = np.squeeze(warped, axis=0)
    else:
        pass
    warped = warped.astype(np.uint8)

    cv2.imwrite(output_path, warped)
    return warped 
