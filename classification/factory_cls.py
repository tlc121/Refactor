import tensorflow as tf
import sys
import os
sys.path.append('../')
#print (os.getcwd())
import os
from classi_nets import resnet_34, resnet_34_dilated, resnet_18, vgg, resnet_50, resnet_101, resnet_152, Densenet_121, Densenet_169, Densenet_201, Densenet_264, MobileNet_V1, MobileNet_V2, Efficient_b0, resnet_18_triplet
import numpy as np



class backbone(object):
    def __init__(self, model, input_data, trainable, classes, keep_prob):
        self.backbone = model
        if self.backbone is 'vgg':
            self.mode = vgg.Vgg(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'res18':
            self.mode = resnet_18.Resnet18(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'res34':
            self.mode = resnet_34.Resnet34(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'res34_dilated':
            self.mode = resnet_34_dilated.Resnet34_dilated(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'res50':
            self.mode = resnet_50.Resnet50(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'res101':
            self.mode = resnet_101.Resnet101(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'res152':
            self.mode = resnet_152.Resnet152(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'dense121':
            self.mode = Densenet_121.Densenet121(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'dense169':
            self.mode = Densenet_169.Densenet169(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'dense201':
            self.mode = Densenet_201.Densenet201(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'dense264':
            self.mode = Densenet_264.Densenet264(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'mobilev1':
            self.mode = MobileNet_V1.Mobilenet_v1(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'mobilev2':
            self.mode = MobileNet_V2.Mobilenet_v2(input_data, trainable, classes, keep_prob)
        elif self.backbone is 'efficient_b0':
            self.mode = Efficient_b0.Efficient(input_data, trainable, classes, keep_prob) 

        

    def compute_loss(self, labels):
        return self.mode.compute_loss(labels=labels)


    def predict(self):
        return self.mode.predict()
    
    def cam(self):
        return self.mode.cam()
