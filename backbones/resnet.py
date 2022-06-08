'''ResNet18/34/50/101/152 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)

	def forward(self, x):

		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	# expansion = 4
	expansion = 2  # to accord with code's downsample

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		# self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.conv1 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)
		# self.bn1 = nn.BatchNorm2d(planes)
		self.bn1 = nn.BatchNorm2d(in_planes)
		# self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False)
		# self.bn2 = nn.BatchNorm2d(planes)
		self.bn2 = nn.BatchNorm2d(in_planes)
		# self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
		self.conv3 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		# self.bn3 = nn.BatchNorm2d(self.expansion * planes)
		self.bn3 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		# if stride != 1 or in_planes != self.expansion * planes:
		# 	self.shortcut = nn.Sequential(
		# 		nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
		# 		nn.BatchNorm2d(self.expansion * planes)
		# 	)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


model_blocks = {
	18: [BasicBlock, [2, 2, 2, 2]],
	34: [BasicBlock, [3, 4, 6, 3]],  # basicblock
	50: [Bottleneck, [3, 4, 6, 3]],  # bottleneck
	101: [Bottleneck, [3, 4, 23, 3]],
	152: [Bottleneck, [3, 8, 36, 3]]
}


class Backbone(nn.Module):
	"""
	   Class for DarknetSeg. Subclasses PyTorch's own "nn" module
	"""

	def __init__(self, params):
		super(Backbone, self).__init__()
		self.use_range = params["input_depth"]["range"]
		self.use_xyz = params["input_depth"]["xyz"]
		self.use_remission = params["input_depth"]["remission"]
		self.drop_prob = params["dropout"]
		self.bn_d = params["bn_d"]
		self.OS = params["OS"]
		self.layers = params["extra"]["layers"]
		print(f"Using {params['name']}" + str(self.layers) + " Backbone")

		# input depth calc
		self.input_depth = 0
		self.input_idxs = []
		if self.use_range:
			self.input_depth += 1
			self.input_idxs.append(0)
		if self.use_xyz:
			self.input_depth += 3
			self.input_idxs.extend([1, 2, 3])
		if self.use_remission:
			self.input_depth += 1
			self.input_idxs.append(4)
		print("Depth of backbone input = ", self.input_depth)

		# stride play
		self.strides = [2, 2, 2, 2]
		# check current stride
		current_os = 1
		for s in self.strides:
			current_os *= s
		print("Original OS: ", current_os)

		# make the new stride
		if self.OS > current_os:
			print("Can't do OS, ", self.OS,
				  " because it is bigger than original ", current_os)
		else:
			# redo strides according to needed stride
			for i, stride in enumerate(reversed(self.strides), 0):
				if int(current_os) != self.OS:
					if stride == 2:
						current_os /= 2
						self.strides[-1 - i] = 1
					if int(current_os) == self.OS:
						break
			print("New OS: ", int(current_os))
			print("Strides: ", self.strides)

		# check that darknet exists
		assert self.layers in model_blocks.keys()

		# generate layers depending on darknet type
		self.blocks = model_blocks[self.layers][1]

		# input layer
		self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
							   stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(32, momentum=self.bn_d)
		self.relu1 = nn.LeakyReLU(0.1)

		# encoder
		block_name = model_blocks[self.layers][0]
		self.enc1 = self._make_enc_layer(block_name, [32, 64], self.blocks[0],
										 stride=self.strides[0], bn_d=self.bn_d)
		self.enc2 = self._make_enc_layer(block_name, [64, 128], self.blocks[1],
										 stride=self.strides[1], bn_d=self.bn_d)
		self.enc3 = self._make_enc_layer(block_name, [128, 256], self.blocks[2],
										 stride=self.strides[2], bn_d=self.bn_d)
		self.enc4 = self._make_enc_layer(block_name, [256, 512], self.blocks[3],
										 stride=self.strides[3], bn_d=self.bn_d)

		# for a bit of fun
		self.dropout = nn.Dropout2d(self.drop_prob)

		# last channels
		self.last_channels = 512

	def _make_enc_layer(self, block, planes, blocks, stride, bn_d=0.1):
		layers = []

		#  downsample
		layers.append(("conv", nn.Conv2d(planes[0], planes[1],
										 kernel_size=3,
										 stride=[1, stride], dilation=1,
										 padding=1, bias=False)))
		layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
		layers.append(("relu", nn.LeakyReLU(0.1)))

		#  blocks
		inplanes = planes[1]
		for i in range(0, blocks):
			layers.append(("residual_{}".format(i), block(*planes, bn_d)))

		return nn.Sequential(OrderedDict(layers))

	def run_layer(self, x, layer, skips, os):
		y = layer(x)
		if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
			skips[os] = x.detach()
			os *= 2
		x = y
		return x, skips, os

	def forward(self, x):
		# filter input
		x = x[:, self.input_idxs]

		# run cnn
		# store for skip connections
		skips = {}
		os = 1

		# first layer
		x, skips, os = self.run_layer(x, self.conv1, skips, os)
		x, skips, os = self.run_layer(x, self.bn1, skips, os)
		x, skips, os = self.run_layer(x, self.relu1, skips, os)

		# all encoder blocks with intermediate dropouts
		x, skips, os = self.run_layer(x, self.enc1, skips, os)
		x, skips, os = self.run_layer(x, self.dropout, skips, os)
		x, skips, os = self.run_layer(x, self.enc2, skips, os)
		x, skips, os = self.run_layer(x, self.dropout, skips, os)
		x, skips, os = self.run_layer(x, self.enc3, skips, os)
		x, skips, os = self.run_layer(x, self.dropout, skips, os)
		x, skips, os = self.run_layer(x, self.enc4, skips, os)
		x, skips, os = self.run_layer(x, self.dropout, skips, os)

		return x, skips

	def get_last_depth(self):
		return self.last_channels

	def get_input_depth(self):
		return self.input_depth
