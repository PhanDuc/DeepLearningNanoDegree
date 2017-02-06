## Application 1: Style Transfer

- Git Repo: [fast style transfer](https://github.com/lengstrom/fast-style-transfer)
- Some styles: [Style View](https://github.com/lengstrom/fast-style-transfer/tree/master/examples/style)
- Style Checkpoint Files: [Google Drive](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ)

### Setup Environment 

OS X and Linux
For OS X and Linux, you'll need to install TensorFlow 0.11.0, Python 2.7.9, Pillow 3.4.2, scipy 0.18.1, and numpy 1.11.2.

In your terminal, enter this commands line by line:

```
conda create -n style-transfer python=2.7.9
source activate style-transfer
conda install -c conda-forge tensorflow=0.11.0
conda install scipy pillow
```

### Transferring styles

1. Download the Zip archive from the fast-style-transfer repository and extract it. You can download it by clicking on the bright green button on the right.
2. Download the Rain Princess checkpoint from here. Put it in the fast-style-transfer folder. A checkpoint file is a model that already has tuned parameters. By using this checkpoint file, we won't need to train the model and can get straight to applying it.
3. Copy the image you want to style into the fast-style-transfer folder.
4. Enter the Conda environment you created above, if you aren't still in it.
5. In your terminal, navigate to the fast-style-transfer folder and enter
```
python evaluate.py --checkpoint ./rain-princess.ckpt --in-path <path_to_input_file> --out-path ./output_image.jpg
```

> Note: Your checkpoint file might be names rain_princess.ckpt, notice the underscore, it's not the dash from above.

## Application 2: Deep Traffic

- [DeepTraffic simulator](http://selfdrivingcars.mit.edu/deeptrafficjs/)
- To learn more about setting the parameters and training the network, read the [overview here](http://selfdrivingcars.mit.edu/deeptraffic/).

## Application 3: Flappy Bird

- [Yenchen Lin's Github Repo](https://github.com/yenchenlin/DeepLearningFlappyBird)

### Instructions
1. Install miniconda or anaconda if you have not already. You can follow our tutorial for help.
2. Create an environment for flappybird
    - Mac/Linux: `conda create --name=flappybird python=2.7`
    - Windows: `conda create --name=flappybird python=3.5`
3. Enter your conda environment
    - Mac/Linux: `source activate flappybird`
    - Windows: `activate flappybird`
4. `conda install -c menpo opencv3`
5. `pip install pygame`
6. `pip install tensorflow`
7. `git clone https://github.com/yenchenlin/DeepLearningFlappyBird.git`
8. `cd DeepLearningFlappyBird`
9. `python deep_q_network.py`

If all went correctly, you should be seeing a Deep Learning based agent play Flappy Bird! The repository contains instructions for training your own agent if you're interested!