import numpy as np
import cv2
import matplotlib.pyplot as plt
from config_seg import cfg
import tensorflow as tf
import random
import os
from multiprocessing.pool import ThreadPool


#dataset iterator
class Dataset(object):
    def __init__(self,num_classes, dataset_type):
        self.batchsize = cfg.TRAIN.BATCHSIZE
        self.num_classes = num_classes
        self.dataaug = cfg.TRAIN.DATAAUG if dataset_type == 'train' else cfg.TEST.DATAAUG
        self.anno_path_ori_img = cfg.TRAIN.ORI_IMG if dataset_type == 'train' else cfg.TEST.ORI_IMG
        #self.anno_path_mask = cfg.TRAIN.MASK_IMG if dataset_type == 'train' else cfg.TEST.MASK_IMG
        self.mask_path = cfg.TRAIN.MASK_PATH if dataset_type == 'train' else cfg.TEST.MASK_PATH
        self.inputsize = cfg.TRAIN.INPUTSIZE if dataset_type == 'train' else cfg.TEST.INPUTSIZE
        self.annotations_ori_img = self.load_annotations()
        self.num_samples = len(self.annotations_ori_img)
        self.augmentation_list = cfg.TRAIN.AUGMENTATION
        self.num_batches = self.num_samples // self.batchsize
        self.pool = ThreadPool(processes=4)
        self.image_batch = []
        self.batch_count = 0

    def load_annotations(self):
        with open(self.anno_path_ori_img, 'r') as f:
            txt = f.readlines()
            annotations_ori_img = [line for line in txt]
        return annotations_ori_img

    def parse_annotations(self, annotations_ori_img, dataaug, idx):
        line_ori_img = annotations_ori_img[idx]
        try:
            ori_img_path = line_ori_img.strip()
        except Exception as e:
            print (e, annotations_ori_img[idx])
            
        img_name = ori_img_path.split('/')[-1].split('.')[0]
        
        if ori_img_path.split('.')[1] == 'npy':
            ori_image = np.load(ori_img_path)
        else:
            ori_image = cv2.imread(ori_img_path)
        
        mask_path = os.path.join(self.mask_path, img_name + '.npy')
        mask = np.load(mask_path)

        if dataaug == True:
            try:
                if 'flip' in self.augmentation_list:
                    ori_image, mask = self.random_flip(ori_image, mask)
                if 'rotate' in self.augmentation_list:
                    ori_image, mask = self.random_rotation(ori_image, mask)
                if 'hsv' in self.augmentation_list:
                    ori_image, mask = self.random_hsv_transform(ori_image, mask, 10, 0.1, 0.1)
#                 if 'crop' in self.augmentation_list:
#                     ori_image, mask = self.random_crop(ori_image, mask, 0.8, 0.1)
                if 'translation' in self.augmentation_list:
                    ori_image, mask = self.image_translation(ori_image, mask)
            except Exception as e:
                print (e, img_path)
        try:
            ori_image, mask = self.img_resize(ori_image, mask, (cfg.TRAIN.INPUTSIZE[0], cfg.TRAIN.INPUTSIZE[1]))
            ori_image, mask = self.im_norm(ori_image, mask)
        except Exception as e:
            print (e, ori_img_path, mask_path)
        
        return ori_image, mask


    def __iter__(self):
        return self


    def next(self):
        if self.batch_count < self.num_batches:
            self.image_batch = []
            batch_image = np.zeros((self.batchsize, self.inputsize[0], self.inputsize[1], 3), dtype=np.float32)
            batch_label = np.zeros((self.batchsize, self.inputsize[0], self.inputsize[1], self.num_classes), dtype=np.float32)
            for i in range(self.batchsize*self.batch_count, self.batchsize*(self.batch_count+1)):
                self.image_batch.append(self.parse_annotations(self.annotations_ori_img, self.dataaug, i))

            for num, img in enumerate(self.image_batch):
                image, label = img[0], img[1]
                batch_image[num, :, :, :] = image
                batch_label[num, :, :, :] = label

            self.batch_count += 1
            return batch_image, batch_label

        else:
            self.batch_count = 0
            np.random.shuffle(self.annotations_ori_img)
            raise StopIteration
                    
###############################################################################################################################################3                    
#data augmentation part

    def random_flip(self, ori_image, mask):
        random_num = random.randint(1, 4)
        return cv2.flip(ori_image, random_num), cv2.flip(mask, random_num)
    
    def random_rotation(self,ori_image, mask):
        random_num = random.randint(-15,15)
        if random_num == 1:
            angle = 90
        elif random_num == 2:
            angle = 180
        elif random_num == 3:
            angle = 270
        return self.rotate_bound(ori_image, random_num), self.rotate_bound(mask, random_num)
        
        
    def random_noise(self, image):
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        
    def rotate_bound(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))
    
    def image_translation(self, ori_image, mask):
        row, col, _ = image.shape
        x = random.randint(int(-col*0.1), int(col*0.1))
        y = random.randint(int(-row*0.1), int(row*0.1))
        M = np.float32([[1,0,x],[0,1,y]])
        ori_image = cv2.warpAffine(ori_image, M, (ori_image.shape[1], ori_image.shape[0])) #11
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        return ori_image, mask   
    
    def hsv_transform(self, img, hue_delta, sat_mult, val_mult):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float)
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        img_hsv[:, :, 1] *= sat_mult
        img_hsv[:, :, 2] *= val_mult
        img_hsv[img_hsv > 255] = 255
        return cv2.cvtColor(np.round(img_hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)

    def random_hsv_transform(self, img, hue_vari, sat_vari, val_vari):
        hue_delta = np.random.randint(-hue_vari, hue_vari)
        sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
        val_mult = 1 + np.random.uniform(-val_vari, val_vari)
        return self.hsv_transform(img, hue_delta, sat_mult, val_mult)
    
    def random_crop(self, image, area_ratio, hw_vari):
        crop_image = lambda img, x0, y0, w, h: img[y0:y0+h, x0:x0+w]
        h, w = image.shape[:2]
        hw_delta = np.random.uniform(-hw_vari, hw_vari)
        hw_mult = 1 + hw_delta

        w_crop = int(round(w*np.sqrt(area_ratio*hw_mult)))

        if w_crop > w:
            w_crop = w

        h_crop = int(round(h*np.sqrt(area_ratio/hw_mult)))
        if h_crop > h:
            h_crop = h

        x0 = np.random.randint(0, w-w_crop+1)
        y0 = np.random.randint(0, h-h_crop+1)
        return crop_image(image, x0, y0, w_crop, h_crop)

  
    def im_norm(self, ori_image, mask):
        ori_img = ori_image/255.
        mask = mask/255.
        return ori_img, mask
    
    def img_resize(self, ori_image, mask, target_size):
        nh, nw = target_size
        ori_image_resized = np.zeros((self.inputsize[0], self.inputsize[1], 3), dtype=np.float32)
        mask_resized = np.zeros((self.inputsize[0], self.inputsize[1], self.num_classes), dtype=np.float32)
        ori_image_resized = cv2.resize(ori_image, (nw, nh))
        for i in range(self.num_classes):
            mask_resized[..., i] = cv2.resize(mask[..., i], (nw, nh))
        return ori_image_resized, mask_resized


    def __len__(self):
        return self.num_batches
