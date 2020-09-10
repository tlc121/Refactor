from easydict import EasyDict as edict

__C  = edict()

cfg = __C

__C.INITIALIZER = 'normal' ##'xavier', 'normal', 'uniform', 'truncated'##
__C.SCALE = 1.0
__C.ACTIVATION = 'leaky_relu' #'relu', 'swish', 'sigmoid', 'tanh', 'leaky_relu', 'h-swish'