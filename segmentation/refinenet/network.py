from encoder_bk import encoder
from decoder_bk import decoder


def network(input_, trainable):
    route_0, route_1, route_2, route_3, route_4 = encoder(input_, trainable)
    output_ = decoder(route_0, route_1, route_2, route_3, route_4, trainable)
    return output_