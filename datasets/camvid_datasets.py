import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import json 
import os 
import torch 
from pathlib import Path
import icecream as ic 

from .utils import * 
import _init_path
import configs.constants as C 


class Camvid(data.Dataset):
    def __init__(self, root, split, is_aug = True, label_index = None, \
                num_classes = 12, img_size = None, \
                class_weights = None,
                check = True):
        self.root = root 
        assert split in ('train', 'val', 'test')
        self.split = split 
        self.labeled_index = label_index

        self.img_size = img_size
        self.imgs = json.load(open(C.data_split_path))[split]['labeled']
        self.labels = list(map(lambda x: x.replace(split, split + 'annot'), self.imgs))

        self.is_aug = is_aug 
        self.num_classes = num_classes
        self.check = check

        if label_index:
            self.imgs = [self.imgs[i] for i in self.labeled_index]
        self.class_weights = class_weights if class_weights != None else self.calculate_weights() 

    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, index: int):
        img_path = os.path.join(self.root, self.imgs[index])
        assert os.path.exists(img_path), 'file {} not found'.format(img_path)
        img = load_img(img_path, crop_size = (int(self.img_size[0]), int(self.img_size[1])))
        


        label_path =  os.path.join(self.root,self.labels[index])

        assert os.path.exists(label_path), 'file {} not found'.format(label_path)

        label = load_label(label_path, crop_size = (int(self.img_size[0]), int(self.img_size[1])))
        assert max(np.unique(label)) < self.num_classes, 'label value exceeds num_classes'

        if self.is_aug:
            transformed = heavy_transform(crop_size = self.img_size)(image = img, label = label)
            img = transformed['image']
            label = transformed['label']


        if self.check:
            os.makedirs(os.path.dirname(os.path.join(C.save_check, self.imgs[index])), exist_ok = True)

            rgb_label = id_to_rgb(label, num_classes= 12)
            cv2.imwrite(os.path.join(C.save_check,  self.imgs[index].split('.')[0] + '_label.png'), np.array(rgb_label))
            cv2.imwrite(os.path.join(C.save_check, self.imgs[index].split('.')[0] + '_img.png'), np.array(img))



        np_img = basic_transform(img)
        np_label = label.transpose(-1, -2).astype('int32')
        
        return np_img, np_label, self.imgs[index]
    

    def calculate_weights(self):
        count = np.ones((self.num_classes, ))
        for img, label in zip(self.imgs, self.labels):
            if len(img) <= 1:
                raise f'lack of rgb or labelling path for {img}'
            label_path = os.path.join(self.root, label)
            label = load_label(label_path, crop_size = self.img_size)
            for i in range(self.num_classes):
                count[i] += np.sum(label == i)
        print('number of pixels in cur classes:', count)
        count = np.log(count)
        cw = count/np.sum(count)

        cw = np.where(cw >0, 1/cw, 0)

        factor = np.ones(self.num_classes)
        factor[2] = 3
        factor[6] = 3
        # factor[7] = 2 
        # factor[9] = 2 
        # factor[10] = 2 
        factor[11] = 2
        
        cw *= factor 
        class_weights = torch.from_numpy(cw / np.sum(cw)).type('torch.FloatTensor')
        print('class_weights', class_weights)

        return class_weights
    



if __name__ == '__main__':
    json_path = C.data_split_path
    with open(json_path) as f:
        data = json.load(f)
        print(data['train']['labeled'][0]) # train/01TP_30_1860/01TP_000030.png
        print(data['val']['labeled'][0])
        print(data['test']['labeled'][0])


    camvid_dataset = Camvid(root = '/home/linlin/dataset/camvid_video', split = 'train', \
                            is_aug = True, label_index = None, num_classes = 12, \
                            img_size = [256, 256], class_weights= None, check = True)
    print(len(camvid_dataset))

    for i in range(1):
        img, label, _ = camvid_dataset[i]
        print(img.shape, label.shape)
        print(np.unique(label))
        print(np.max(label))
        print(np.min(label))

        print(img.dtype, label.dtype)
  