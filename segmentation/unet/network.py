from encoder_bk import encoder
from decoder_bk import decoder


def network(input_, trainable):
    route_0, route_1, route_2, route_3, route_4, input_ = encoder(input_, trainable)
    output_ = decoder(route_0, route_1, route_2, route_3, route_4, input_, trainable)
    return output_