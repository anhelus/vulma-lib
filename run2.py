from src.algs.gradcam import GradCAM, visualize

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2
import os


def gradcam(args):
	if not args.verbose:
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
		tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
	model = tf.keras.models.load_model(args.model)
	if args.verbose:
		print('Model loaded.')
	visualize(model, args.image, args.size)
	# img = cv2.imread(args.image)
	# image = load_img(args.image, target_size=args.size)
	# image = img_to_array(image)
	# # Simulate a batch
	# image = np.expand_dims(image, axis=0)
	# image = imagenet_utils.preprocess_input(image)
	# preds = model.predict(image)
	# i = np.argmax(preds[0])
	# cam = GradCAM(model, i)
	# heatmap = cam.compute_heatmap(image)
	# heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
	# (heatmap, output) = cam.overlay_heatmap(heatmap, img, alpha=0.5)
	# output = imutils.resize(output, height=700)
	# cv2.imshow("Output", output)
	# cv2.waitKey(0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-i',
		'--image',
		type=str,
		required=True,
		help='Path to the input image.')
	parser.add_argument(
		'-m',
		'--model',
		type=str,
		required=True,
		help='Pre-trained model to be used.')
	parser.add_argument(
		'-v',
		'--verbose',
		type=bool,
		default=False,
		help='Set verbosity.')
	parser.add_argument(
		'-s',
		'--size',
		default=(224, 224),
		help='Image size.')
	args = parser.parse_args()
	gradcam(args)

