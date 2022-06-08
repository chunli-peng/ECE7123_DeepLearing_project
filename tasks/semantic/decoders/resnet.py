import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


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
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		# self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.conv1 = nn.Conv2d(planes, in_planes, kernel_size=1, bias=False)
		# self.bn1 = nn.BatchNorm2d(planes)
		self.bn1 = nn.BatchNorm2d(in_planes)
		# self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1, bias=False)
		# self.bn1 = nn.BatchNorm2d(planes)
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


class Decoder(nn.Module):
	def __init__(self, params, stub_skips, network_layers, OS=16, feature_depth=512):
		super(Decoder, self).__init__()
		self.backbone_OS = OS
		self.backbone_feature_depth = feature_depth
		self.drop_prob = params["dropout"]
		self.bn_d = params["bn_d"]

		# stride play
		self.strides = [2, 2, 2, 2]
		# check current stride
		current_os = 1
		for s in self.strides:
			current_os *= s
		print("Decoder original OS: ", int(current_os))
		# redo strides according to needed stride
		for i, stride in enumerate(self.strides):
			if int(current_os) != self.backbone_OS:
				if stride == 2:
					current_os /= 2
					self.strides[i] = 1
				if int(current_os) == self.backbone_OS:
					break
		print("Decoder new OS: ", int(current_os))
		print("Decoder strides: ", self.strides)

		# decoder
		block = BasicBlock if network_layers <= 34 else Bottleneck
		self.dec4 = self._make_dec_layer(block, [512, 256], bn_d=self.bn_d,
										 stride=self.strides[0])
		self.dec3 = self._make_dec_layer(block, [256, 128], bn_d=self.bn_d,
										 stride=self.strides[1])
		self.dec2 = self._make_dec_layer(block, [128, 64], bn_d=self.bn_d,
										 stride=self.strides[2])
		self.dec1 = self._make_dec_layer(block, [64, 32], bn_d=self.bn_d,
										 stride=self.strides[3])

		# layer list to execute with skips
		self.layers = [self.dec4, self.dec3, self.dec2, self.dec1]

		# for a bit of fun
		self.dropout = nn.Dropout2d(self.drop_prob)

		# last channels
		self.last_channels = 32

	def _make_dec_layer(self, block, planes, bn_d=0.1, stride=2):
		layers = []

		#  downsample
		if stride == 2:
			layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
														kernel_size=[1, 4], stride=[1, 2],
														padding=[0, 1])))
		else:
			layers.append(("conv", nn.Conv2d(planes[0], planes[1],
											 kernel_size=3, padding=1)))
		layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
		layers.append(("relu", nn.LeakyReLU(0.1)))

		#  blocks
		layers.append(("residual", block(*planes, bn_d)))

		return nn.Sequential(OrderedDict(layers))

	def run_layer(self, x, layer, skips, os):
		feats = layer(x)  # up
		if feats.shape[-1] > x.shape[-1]:
			os //= 2  # match skip
			feats = feats + skips[os].detach()  # add skip
		x = feats
		return x, skips, os

	def forward(self, x, skips):
		os = self.backbone_OS

		# run layers
		x, skips, os = self.run_layer(x, self.dec4, skips, os)
		x, skips, os = self.run_layer(x, self.dec3, skips, os)
		x, skips, os = self.run_layer(x, self.dec2, skips, os)
		x, skips, os = self.run_layer(x, self.dec1, skips, os)

		x = self.dropout(x)

		return x

	def get_last_depth(self):
		return self.last_channels
