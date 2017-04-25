#  Autoencoder

See notebook for details

## Simplest Architecture

input layer (encoder) -> hidden layer(RELU) -> output layer (decoder) (Sigmoid)

Information loss depends on the number of nodes in the hidden layer

## Convolution autoencoders

use convolution stack 

- Encoder
    - input -> (conv(stride=1)+max_pool) x 3 (finally arrive at compressed representation)
- Decoder 
    - (Upsample + Conv) x 3 + output

Upsample: deconvolution