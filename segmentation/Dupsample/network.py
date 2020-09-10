from encoder_bk import encoder
from decoder_bk import decoder


def network(input_, trainable):
    out = encoder(input_, trainable)
    output_ = decoder(out, trainable)
    return output_