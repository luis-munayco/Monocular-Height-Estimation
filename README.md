## Description
This project aims to develop a ML model to estimate the height using only monocular satellite images as inputs. To solve this 2D regresion problem,a modified version of the well-know U-Net NN is being used. The original U-Net network was designed to solve classification problems, however, to solve our height regression problem, output activation layer had been modified in order to have continues values in the output.
Likewise, the loss function used in the backpropagation for the U-Net NN had to be also modified. Originally, loss function was set as Dice Score, however, for regression problems, an L1 loss function has to be used.
This model was trained from scratch with 2k images and scored a Mean Absolute Error of 2.25m on over 200 test images.

![Pixel Height Histogram](https://github.com/luis-munayco/Monocular-Height-Estimation/blob/master/imgs/description.png)

## Data

The input images and target masks should be in the `data/train_dataset/imgs` and `data/train_dataset/masks` folders respectively.
Train images dataset contain the satellite monocular images that will be used as input for the model. And train masks dataset contain the height masks that will be used as targets for training the model.

![Prediction1](https://github.com/luis-munayco/Monocular-Height-Estimation/blob/master/imgs/Predicted_1_nb.png)

![Prediction2](https://github.com/luis-munayco/Monocular-Height-Estimation/blob/master/imgs/Predicted_2_nb.png)

However, as it can be seen from the histogram below, most of the masks pixels have low heights values (< 50m). This is expected given that, as in the real world, most of the building have a low/medium height, and skycrapper or tall buildings are mostly excepctions. Having said this, the main challenge to develop an accurate model is the highly imbalanced dataset.

![Pixel Height Histogram](https://github.com/luis-munayco/Monocular-Height-Estimation/blob/master/imgs/histogram.png)

### Training

```console
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks
```
To overcome the problem of the highly imbalanced dataset, a balanced U-Net model have to be designed. This is, weights is going to be applied to the loss function according to the histogram of heights. To calculate the weights, a square inverse function is being used following [Lang, 2003](https://www.nature.com/articles/s41559-023-02206-6)

![Modified U-Net](https://github.com/luis-munayco/Monocular-Height-Estimation/blob/master/imgs/balancedunet.png)

Model was trained with 2k images and scored a Mean Absolute Error of 2.25 on over 200 test images.
Dataset 

### Prediction

After training the model and saving it to `MODEL.pth`, this can be tested with the test masks on the test images.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

```console
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images
```
The model to be used for the prediction can be specified using the argument `--model MODEL.pth`.

![Prediction1nb](https://github.com/luis-munayco/Monocular-Height-Estimation/blob/master/imgs/Predicted_1.png)

![Prediction2nb](https://github.com/luis-munayco/Monocular-Height-Estimation/blob/master/imgs/Predicted_2.png)