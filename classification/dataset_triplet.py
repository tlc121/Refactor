import numpy as np
import cv2
import matplotlib
from config_cls import cfg
import tensorflow as tf
import random
from multiprocessing.pool import ThreadPool
import torch
from random import choice


#dataset iterator
class Dataset(object):
    def __init__(self,num_classes, dataset_type):
        self.batchsize = cfg.TRAIN.BATCHSIZE
        self.num_classes = num_classes
        self.dataset_type = dataset_type
        self.dataaug = cfg.TRAIN.DATAAUG if dataset_type == 'train' else cfg.TEST.DATAAUG
        self.anno_path = cfg.TRAIN.ANNO_PATH if dataset_type == 'train' else cfg.TEST.ANNO_PATH
        self.inputsize = cfg.TRAIN.INPUTSIZE if dataset_type == 'train' else cfg.TEST.INPUTSIZE
        self.train_pos = torch.load('/home/tanglc/TCT_det/data_cls/train_pos.pth')
        self.train_neg = torch.load('/home/tanglc/TCT_det/data_cls/train_neg.pth')
        self.val_pos = torch.load('/home/tanglc/TCT_det/data_cls/val_pos.pth')
        self.val_neg = torch.load('/home/tanglc/TCT_det/data_cls/val_neg.pth')
        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.augmentation_list = cfg.TRAIN.AUGMENTATION
        self.num_batches = self.num_samples // self.batchsize
        self.pool = ThreadPool(processes=4)
        self.image_batch = []
        self.batch_count = 0

    def load_annotations(self):
        with open(self.anno_path, 'r') as f:
            txt = f.readlines()
            annotations = [line for line in txt]
        np.random.shuffle(annotations)
        return annotations

    def parse_annotations(self, annotations, dataaug, idx):
        line = annotations[idx]
        try:
            img_path = line.split(' ')[0]
            label = float(line.split(' ')[1])
            img_name = img_path.split('/')[-2]

        except Exception as e:
            print (e, annotations[idx])
        
        
        if label == 1 and self.dataset_type == 'train':
            img_pos_path = choice(self.train_pos)
            img_neg_path = choice(self.train_neg)
        elif label == 0 and self.dataset_type == 'train':
            img_pos_path = choice(self.train_neg)
            img_neg_path = choice(self.train_pos)
        elif label == 0 and self.dataset_type == 'test':
            img_pos_path = choice(self.val_neg)
            img_neg_path = choice(self.val_pos)
        elif label == 1 and self.dataset_type == 'test':
            img_pos_path = choice(self.val_pos)
            img_neg_path = choice(self.val_neg)
            
        
        
        if img_path.split('.')[1] == 'npy':
            image = np.load(img_path)
            image_pos = np.load(img_pos_path)
            image_neg = np.load(img_neg_path)
        else:
            image = cv2.imread(img_path)
            image_pos = cv2.imread(img_pos_path)
            image_neg = cv2.imread(img_neg_path)

        if dataaug == True:
            try:
                if 'flip' in self.augmentation_list:
                    image = self.random_flip(image)
                    image_pos = self.random_flip(image_pos)
                    image_neg = self.random_flip(image_neg)
                if 'rotate' in self.augmentation_list:
                    image = self.random_rotation(image)
                if 'hsv' in self.augmentation_list:
                    image = self.random_hsv_transform(image, 10, 0.1, 0.1)
                    image_pos = self.random_hsv_transform(image_pos, 10, 0.1, 0.1)
                    image_neg = self.random_hsv_transform(image_neg, 10, 0.1, 0.1)
                if 'crop' in self.augmentation_list:
                    image = self.random_crop(image, 0.9, 0.1)
                    image_pos = self.random_crop(image_pos, 0.9, 0.1)
                    image_neg = self.random_crop(image_neg, 0.9, 0.1)
                if 'translation' in self.augmentation_list:
                    image = self.image_translation(image)
                    image_pos = self.image_translation(image_pos)
                    image_neg = self.image_translation(image_neg)
            except Exception as e:
                print (e, img_path)
        try:
            image = self.img_resize(image, (cfg.TRAIN.INPUTSIZE, cfg.TRAIN.INPUTSIZE))
            image_pos = self.img_resize(image_pos, (cfg.TRAIN.INPUTSIZE, cfg.TRAIN.INPUTSIZE))
            image_neg = self.img_resize(image_neg, (cfg.TRAIN.INPUTSIZE, cfg.TRAIN.INPUTSIZE))
        except Exception as e:
            print (e, img_path)
        image = self.im_norm(image)
        image_pos = self.im_norm(image_pos)
        image_neg = self.im_norm(image_neg)
        return image, image_pos, image_neg


    def __iter__(self):
        return self


    def next(self):
        if self.batch_count < self.num_batches:
            self.image_batch = []
            batch_image_anchor = np.zeros((self.batchsize, self.inputsize, self.inputsize, 3), dtype=np.float32)
            batch_image_postive = np.zeros((self.batchsize, self.inputsize, self.inputsize, 3), dtype=np.float32)
            batch_image_negative = np.zeros((self.batchsize, self.inputsize, self.inputsize, 3), dtype=np.float32)
            batch_label = np.zeros((self.batchsize), dtype=np.float32)
            for i in range(self.batchsize*self.batch_count, self.batchsize*(self.batch_count+1)):
                self.image_batch.append(self.parse_annotations(self.annotations, self.dataaug, i))

            for num, img in enumerate(self.image_batch):
                image, image_pos, image_neg = img[0], img[1], img[2]
                batch_image_anchor[num, :, :, :] = image
                batch_image_postive[num, :, :, :] = image_pos
                batch_image_negative[num, :, :, :] = image_neg
                #batch_label[num] = label
                #label smoothing
#                 uniform_distribution = np.full(self.num_classes, 1.0/self.num_classes)
#                 deta = 0.05
#                 batch_label = batch_label * (1 - deta) + deta * uniform_distribution

            self.batch_count += 1
            return batch_image_anchor, batch_image_postive, batch_image_negative#, batch_label

        else:
            self.batch_count = 0
            np.random.shuffle(self.annotations)
            raise StopIteration
                    
###############################################################################################################################################3                    
#data augmentation part

    def random_flip(self, image):
        random_num = random.randint(1, 4)
        return cv2.flip(image, random_num)
    
    def random_rotation(self,image):
        random_num = random.randint(-15,15)
        if random_num == 1:
            angle = 90
        elif random_num == 2:
            angle = 180
        elif random_num == 3:
            angle = 270
        return self.rotate_bound(image, random_num)
        
        
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
    
    def image_translation(self, image):
        row, col, _ = image.shape
        x = random.randint(int(-col*0.1), int(col*0.1))
        y = random.randint(int(-row*0.1), int(row*0.1))
        M = np.float32([[1,0,x],[0,1,y]])
        shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0])) #11
        return shifted
    
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

  
    def im_norm(self, image):
        img = image/255.
        return img
    
    def img_resize(self, image, target_size):
        ih, iw    = target_size
        h,  w, _  = image.shape

        scale = min(float(iw)/w, float(ih)/h)
        nw, nh  = int(scale * w), int(scale * h)
        try:
            image_resized = cv2.resize(image, (nw, nh), interpolation=1)
        except Exception as e:
            print (str(e))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=0)
        dw, dh = (iw - nw) // 2, (ih-nh) // 2
        image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
        return image_paded


    def __len__(self):
        return self.num_batches
