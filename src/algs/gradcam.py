from tensorflow.keras.models import Model
# from tensorflow.keras.engine.functional import Functional
# from tensorflow.keras.engine.sequential import Sequential
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.preprocessing.image import (img_to_array, load_img)
from tensorflow.keras.applications import imagenet_utils

import imutils


class GradCAM():
    """ Implements the GradCAM algorithm.
    """

    def __init__(
        self,
        model,
        class_idx,
        layer_name=None) -> None:
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
    
    @property
    def model(self):
        return self.__model
    
    @model.setter
    def model(self, value):
        # if not (isinstance(value, Functional) \
        #     or (isinstance(value, Sequential))):
        #     raise ValueError('Model should be either a Functional or a Sequential model.')
        self.__model = value
    
    @property
    def class_idx(self):
        return self.__class_idx
    
    @class_idx.setter
    def class_idx(self, value):
        self.__class_idx = value

    @property
    def layer_name(self):
        return self.__layer_name
    
    @layer_name.setter
    def layer_name(self, value):
        if value is None:
            self.__layer_name = self.find_target_layer()    # TODO
        else:
            self.__layer_name = value
    
    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check if the layer has a 4d output
            if len(layer.output_shape) == 4:
                return layer.name
        
        # otherwise, we could not find a 4D layer so the gradcam
        # algorithm cannot be applied
        raise ValueError('Could not find 4D layer. Cannot apply GradCAM.')
    

    def compute_heatmap(self, image, eps=1e-8):
        """ construct our gradient model by supplying
        1. inputs to our pretrained model
        2. output of the presumably final 4d layer in the network
        3. output of softmax activations from the model
        """
        grad_model = Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.layer_name).output,
                self.model.output]
        )
        # record operation for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass
            # the image throught the gradient model,
            # grab the loss associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (conv_outputs, predictions) = grad_model(inputs)
            loss = predictions[:, self.class_idx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, conv_outputs)
        # compute the guided gradients
        cast_conv_outputs = tf.cast(conv_outputs > 0, 'float32')
        cast_grads = tf.cast(grads > 0, 'float32')
        guided_grads = cast_conv_outputs * cast_grads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        conv_outputs = conv_outputs[0]
        guided_grads = guided_grads[0]
        # essentially what we are doing here is finding positive values
        # of both cast_conv_outputs and cast_grads, followed by multiplying 
        # them by the gradient of the differentiation - this operation will allow
        # us to visualize where in the volume the network is activating
        # later in the compute_heatmap function
        # now
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

        # grab the spatial dimension of the input image and resize the output 
        # class activation map to match the input image dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range 
        # [0, 1], scale the resulting values to the range [0, 25],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype('uint8')

        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        # apply the supplied color map to the heatmap and then overaly the heatmap on the
        # input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)


def visualize(model, image_path, size, outpath):
    image_path = str(image_path)
    img = cv2.imread(image_path)
    image = load_img(image_path, target_size=size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    preds = model.predict(image)
    i = np.argmax(preds[0])
    cam = GradCAM(model, i)
    heatmap = cam.compute_heatmap(image)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    (heatmap, output) = cam.overlay_heatmap(heatmap, img, alpha=0.5)
    output = imutils.resize(output, height=700)
    cv2.imshow("Output", output)
    cv2.imwrite(outpath, output)
    cv2.waitKey(10)


"""
automatic differentiation -> process of computing a value and computing derivatives of that value

tf2.0 fornisce un'implementazione della automatic differentiation attraverso quello che chiamano
gradient tape:https://www.tensorflow.org/guide/autodiff?hl=en

our inference stops at the specific layer we are concerned about -> we do not need to compute a full
forward pass
"""