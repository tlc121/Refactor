from easydict import EasyDict as edict

__C  = edict()

cfg = __C

#Train Options
__C.TRAIN = edict()

__C.TRAIN.ANNO_PATH = '/home/tanglc/her2_FISCH/dataset/counting/train_anno.txt'
__C.TRAIN.INPUTSIZE = 224
__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.BATCHSIZE = 32
__C.TRAIN.PRETRAIN_MODE = 'backbone' #'backbone' or 'whole'
__C.TRAIN.BACKBONE_PRETRAIN = "/hdd/sd5/tlc/pretrain_model/darknet53/yolov3_test_loss=136.5372.ckpt-80"
__C.TRAIN.INITIAL_WEIGHT = '/hdd/sd5/tlc/OCR/Model_ckpt/vgg_epoch=35_test_loss=0.2374.ckpt-35'
__C.TRAIN.SAVE_PATH_PB  = '/hdd/sd5/tlc/FISCH/Model_pb/counting/'
__C.TRAIN.SAVE_PATH_CKPT  = '/hdd/sd5/tlc/FISCH/Model_ckpt/counting/'
__C.TRAIN.AUGMENTATION = ['flip', 'rotate'] #'flip', 'rotate', 'hsv', 'translation', 'crop'
__C.TRAIN.DATAAUG = True
__C.TRAIN.EPOCH = 200
__C.TRAIN.SAVE = 5
__C.TRAIN.REG_NUM = 2
__C.TRAIN.NETWORK = 'res18'
__C.TRAIN.MOMENTUM = 0.999


#Val Options
__C.TEST = edict()
__C.TEST.ANNO_PATH = '/home/tanglc/her2_FISCH/dataset/counting/val_anno.txt'
__C.TEST.INPUTSIZE = 224
__C.TEST.BATCHSIZE = 1
__C.TEST.NETWORK = 'res18'
__C.TEST.WEIGHT_FILE = '/data/tlc/model_res34/dense169_epoch=5.pb_test_loss=0.9789.ckpt-5'
__C.TEST.DATAAUG = False