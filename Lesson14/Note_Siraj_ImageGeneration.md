#  Siraj's Image Generation: Autoencoder

[Code](https://github.com/llSourcell/how_to_generate_images) for reference

[Live Video for image generation](https://www.youtube.com/watch?v=iz-TZOEKXzA)

## Resources

- [Variational autoencoder tutorial](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
- [Variational autoencoders explained](http://kvfrans.com/variational-autoencoders-explained/)
- [Introducing variational autoencoders](http://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and.html)
- [Under the hood of the variational autoencoder](http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html)
- [Categorical Variational Autoencoders using Gumbel-Softmax](http://blog.evjang.com/2016/11/tutorial-categorical-variational.html)
- [Variational Autoencoder in Tensorflow](https://jmetzen.github.io/2015-11-27/vae.html)


## Autoencoder

Data -> encoder -> decoder -> new data

Autoencoding is Data Compression, but it is quite lose-y

## Application

1. Data Denoising
2. Generate similar but unique data (add smiling, remove smiling, add glasses, remove glasses)

## VAE: variational autoencoder (bayesian inference + deep learning)

Model is stochastic.

Loss function

- *differences between generated image and original image* 
    - generation-loss = mean(square(generated-image - real-image))
- *distribution regularization*
    - latent-loss = KL-Divergence(latent-variable, unit_gaussian)
- Loss = generation_loss + latent_losss

In order to train the model, we **reparameterize** the model



