from easydict import EasyDict as edict

__C  = edict()

cfg = __C

#Train Options
__C.TRAIN = edict()

__C.TRAIN.ORI_IMG = '/home/tanglc/PDL1/dataset/train_ce.txt'
__C.TRAIN.MASK_PATH = '/hdd/sd5/tlc/PDL1/train_mask_ce/'
__C.TRAIN.INPUTSIZE = [512, 512]
__C.TRAIN.LEARN_RATE_INIT = 1e-3
__C.TRAIN.LEARN_RATE_END = 1e-5
__C.TRAIN.BATCHSIZE = 4
__C.TRAIN.PRETRAIN_MODE = 'backbone'
__C.TRAIN.INITIAL_WEIGHT = '/hdd/sd5/tlc/lung_cancer/CAMEL/Model_ckpt/res18_epoch=11_test_loss=0.3286.ckpt-11'
__C.TRAIN.BACKBONE_PRETRAIN = "/ssd2/tlc/pretrain_model/U-net/res18_epoch=29_test_loss=0.6633.ckpt-29"
__C.TRAIN.AUGMENTATION = []
__C.TRAIN.DATAAUG = True
__C.TRAIN.EPOCH = 50
__C.TRAIN.SAVE = 5
__C.TRAIN.BACKBONE = 'unet'
__C.TRAIN.NUMCLASS = 3
__C.TRAIN.MOMENTUM = 0.999


#Val Options
__C.TEST = edict()
__C.TEST.ORI_IMG = '/home/tanglc/PDL1/dataset/val_ce.txt'
__C.TEST.MASK_PATH = '/hdd/sd5/tlc/PDL1/val_mask_ce/'
__C.TEST.INPUTSIZE = [512, 512]
__C.TEST.BATCHSIZE = 4
__C.TEST.NETWORK = 'res18'
__C.TEST.WEIGHT_FILE = '/data/tlc/model_res34/dense169_epoch=5.pb_test_loss=0.9789.ckpt-5'
__C.TEST.DATAAUG = False

