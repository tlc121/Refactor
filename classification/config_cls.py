from easydict import EasyDict as edict

__C  = edict()

cfg = __C

#Train Options
__C.TRAIN = edict()

__C.TRAIN.ANNO_PATH = '/home/tanglc/TCT_det/data_cls/5fold/train_comb.txt'
__C.TRAIN.INPUTSIZE = 224
__C.TRAIN.LEARN_RATE_INIT = 1e-3
__C.TRAIN.LEARN_RATE_END = 1e-5
__C.TRAIN.BATCHSIZE = 128
__C.TRAIN.PRETRAIN_MODE = 'backbone'
__C.TRAIN.INITIAL_WEIGHT = '/hdd/sd5/tlc/OCR/Model_ckpt/vgg_epoch=35_test_loss=0.2374.ckpt-35'
__C.TRAIN.BACKBONE_PRETRAIN = "/ssd2/tlc/pretrain_model/res18/res18_epoch=16_test_loss=0.3583.ckpt-16"
__C.TRAIN.AUGMENTATION = ['flip', 'rotate', 'translation', 'crop', 'hsv']
__C.TRAIN.DATAAUG = True
__C.TRAIN.EPOCH = 50
__C.TRAIN.SAVE = 10
__C.TRAIN.NETWORK = 'res18'
__C.TRAIN.DROPOUT = 0.9
__C.TRAIN.NUMCLASS = 2
__C.TRAIN.MOMENTUM = 0.99


#Val Options
__C.TEST = edict()

__C.TEST.ANNO_PATH = '/home/tanglc/TCT_det/data_cls/val_0515.txt'
__C.TEST.INPUTSIZE = 224
__C.TEST.BATCHSIZE = 64
__C.TEST.WEIGHT_FILE = '/data/tlc/model_res34/dense169_epoch=5.pb_test_loss=0.9789.ckpt-5'
__C.TEST.DATAAUG = False

