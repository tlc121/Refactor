import core.common as common
from config_det import cfg
import tensorflow as tf
from backbone import res34, res50, res101, res18, darknet53, dense121, dense169, dense201, MobileNet_V1, MobileNet_V2


def backbone(input_data, trainable):
    backbone_ = cfg.TRAIN.BACKBONE
    with tf.variable_scope('backbone'):
        if backbone_ == 'res34':
            _, _, _, route_1, route_2, route_3, _, = res34.Res34(input_data, trainable)
        elif backbone_ == 'res18':
            _, _, _, route_1, route_2, route_3, _, = res18.Res18(input_data, trainable)
        elif backbone_ == 'res50':
            _, _, _, route_1, route_2, route_3, _, = res50.Res50(input_data, trainable)
        elif backbone_ == 'res101':
            _, _, _, route_1, route_2, route_3, _, = res101.Res101(input_data, trainable)
        elif backbone_ == 'darknet53':
            route_1, route_2, route_3, _, = darknet53.darknet53(input_data, trainable)
        elif backbone_ == 'dense121':
            route_1, route_2, route_3, _ = dense121.dense121(input_data, trainable)
        elif backbone_ == 'dense169':
            route_1, route_2, route_3, _ = dense169.dense169(input_data, trainable)
        elif backbone_ == 'dense201':
            route_1, route_2, route_3, _ = dense201.dense201(input_data, trainable)
        elif backbone_ == 'MobileNet_V1':
            _, _, _, route_1, route_2, route_3, _, = MobileNet_V1.MobileNet_V1(input_data, trainable)
        elif backbone_ == 'MobileNet_V2':
            _, _, _, route_1, route_2, route_3, _, = MobileNet_V2.MobileNet_V2(input_data, trainable)
        return route_1, route_2, route_3




