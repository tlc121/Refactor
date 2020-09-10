from encoder import encoder
from decoder import decoder

def network(input_, trainable):
    route_ = encoder(input_, trainable)
    output_ = decoder(route_, trainable)
    return output_