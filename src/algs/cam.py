# -*- encode: utf-8
""" This module implements the GradCAM XAI algorithm.
"""
import tensorflow as tf
import matplotlib.cm as cm
import numpy as np

from tensorflow import keras


class GradCAM():

    def __init__(
        self,
        image,
        model):
        self.image = image
        self.model = model
    
    @property
    def image(self):
        return self.__image
    
    @image.setter
    def image(self, value):
        self.__image = value

    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, value):
        self.model = value
    
    def create_heatmap(self, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [
                self.model.get_layer(last_conv_layer_name).output,
                self.model.output
            ]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(self.image)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]



def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index=None):
    """ Create the heatmap to be used for GradCAM evaluation.
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    # Step 1: map input to last conv layer and out.
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            # TODO adapt this to transfer learning models!
            model.get_layer(last_conv_layer_name).output,
            model.output
        ])
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    # Step 2: compute the gradient of the top 
    with tf.GradientTape() as tape:
        
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(
    img_path,
    heatmap,
    save_img: bool = False,
    cam_path: str = "cam.jpg",
    alpha: float = 0.4):
    # Load the original image
    img = keras.preprocessing.image.img_to_array(
        keras.preprocessing.image.load_img(img_path))
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    # jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    if save_img:
        superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))
    # superimposed_img.show()
