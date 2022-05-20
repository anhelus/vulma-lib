from pathlib import Path
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
	if args.path:
		if args.verbose:
			print('Loading images in provided path.')
		images = Path(args.path).glob('*.jpeg')
		for image in images:
			savepath = Path(args.outpath, image)
			print(savepath)
			visualize(model, image, args.size, str(image))
	else:
		print('no here')
		# path = Path(args.outpath, Path(args.image).name)
		# visualize(model, args.image, args.size, str(path))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-i',
		'--image',
		type=str,
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
	parser.add_argument(
		'-o',
		'--outpath',
		default='results',
		help='Output path')
	parser.add_argument(
		'-p',
		'--path',
		help='Path')
	args = parser.parse_args()
	gradcam(args)

