# Copyright (c) <2016> <GUANGHAN NING>. All Rights Reserved.
 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

import numpy as np
import os
import sys
import argparse,logging
import mxnet as mx
import cv2
import time

ctx = mx.cpu(0)

def group(data, num_r, num, kernel, stride, pad, layer):
	if num_r > 0:
		conv_r = mx.symbol.Convolution(data=data, num_filter=num_r, kernel=(1,1), name=('conv%s_r' % layer))
		slice_r = mx.symbol.SliceChannel(data=conv_r, num_outputs=2, name=('slice%s_r' % layer))
		mfm_r = mx.symbol.maximum(slice_r[0], slice_r[1])
		conv = mx.symbol.Convolution(data=mfm_r, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
	else:
		conv = mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num, name=('conv%s' % layer))
	slice = mx.symbol.SliceChannel(data=conv, num_outputs=2, name=('slice%s' % layer))
	mfm = mx.symbol.maximum(slice[0], slice[1])
	pool = mx.symbol.Pooling(data=mfm, pool_type="max", kernel=(2, 2), stride=(2,2), name=('pool%s' % layer))
	return pool


def lightened_cnn_b_feature():
	data = mx.symbol.Variable(name="data")
	pool1 = group(data, 0, 96, (5,5), (1,1), (2,2), str(1))
	pool2 = group(pool1, 96, 192, (3,3), (1,1), (1,1), str(2))
	pool3 = group(pool2, 192, 384, (3,3), (1,1), (1,1), str(3))
	pool4 = group(pool3, 384, 256, (3,3), (1,1), (1,1), str(4))
	pool5 = group(pool4, 256, 256, (3,3), (1,1), (1,1), str(5))
	flatten = mx.symbol.Flatten(data=pool5)
	fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc1")
	slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=2, name="slice_fc1")
	mfm_fc1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
	drop1 = mx.symbol.Dropout(data=mfm_fc1, p=0.7, name="drop1")
	return drop1


def read_img(img_path, size, ctx):
	print('read_img\n')
	print(img_path)
	img_arr = np.zeros((1, 1, size, size), dtype=float)
	img = np.expand_dims(cv2.imread(img_path, 0), axis=0)
	print(img.shape)
	assert(img.shape == (1, size, size))
	img_arr[0][:] = img/255.0
	return img_arr


def cal_dist(output_1, output_2):
	dist = np.dot(output_1, output_2)/np.linalg.norm(output_1)/np.linalg.norm(output_2)
	return dist

def loadFaceRecognitionModel():
	start_time = time.time()
	_, model_args, model_auxs = mx.model.load_checkpoint(os.path.join(os.path.dirname(__file__),'model/lightened_cnn/lightened_cnn'), 166)
	symbol = lightened_cnn_b_feature()
	model_loading_time = time.time() - start_time

	print('Time For Model Loading: ' + str(model_loading_time) + '\n')
	print 'loading face model done'
	return symbol,model_args, model_auxs

def extractFeature(imageFilePath,symbol,model_args, model_auxs):
	start_time = time.time()
	model_args['data'] = mx.nd.array(read_img(imageFilePath, 128, ctx), ctx)
	exector = symbol.bind(ctx, model_args, args_grad=None, grad_req="null", aux_states=model_auxs)
	exector.forward(is_train=False)
	exector.outputs[0].wait_to_read()
	output = exector.outputs[0].asnumpy()
	data_loading_time = time.time() - start_time
	print('Time For Feature Extraction: ' + str(data_loading_time) + '\n')
	return output


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--face_aligned', type=str, default="ning.png",
		        help='The path to the image of an aligned face')
	parser.add_argument('--suffix', type=str, default="png",
		        help='The type of image')
	parser.add_argument('--size', type=int, default=128,
		        help='the image size of the aligned face image, only support square size')
	parser.add_argument('--model-prefix', default='model/lightened_cnn/lightened_cnn',
		        help='The trained model to get feature')
	parser.add_argument('--epoch', type=int, default=166,
		        help='The epoch number of model')
	args = parser.parse_args()

	start_time = time.time()
	_, model_args, model_auxs = mx.model.load_checkpoint(args.model_prefix, args.epoch)
	symbol = lightened_cnn_b_feature()
        model_loading_time = time.time() - start_time  
	print('Time For Model Loading: ' + str(model_loading_time) + '\n')

	start_time = time.time()  
	model_args['data'] = mx.nd.array(read_img(args.face_aligned, args.size, ctx), ctx)
	exector = symbol.bind(ctx, model_args ,args_grad=None, grad_req="null", aux_states=model_auxs)
	data_loading_time = time.time() - start_time  
	print('Time For Data Loading: ' + str(data_loading_time) + '\n')

	start_time = time.time() 
	exector.forward(is_train=False)
	exector.outputs[0].wait_to_read()
	output = exector.outputs[0].asnumpy()
	network_forward_time = time.time() - start_time  
	print('Time For Network Forwarding: ' + str(network_forward_time) + '\n')

	# model_args['data'] = mx.nd.array(read_img('alvin.png', args.size, ctx), ctx)
	# exector = symbol.bind(ctx, model_args ,args_grad=None, grad_req="null", aux_states=model_auxs)
	# exector.forward(is_train=False)
	# exector.outputs[0].wait_to_read()
	# output2 = exector.outputs[0].asnumpy()
	# start_time = time.time()
	# dist= cal_dist(output[0], output2[0])
	# calculate_distance_time = time.time() - start_time
	# print('Time For Calculating Distance: ' + str(calculate_distance_time) + '\n')

	print('\n Feature: ', output)
	# print('\n Distance: ', dist)


if __name__ == '__main__':
	main()
