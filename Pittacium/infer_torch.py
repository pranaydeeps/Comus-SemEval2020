#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

R_MEAN = 123.68
G_MEAN = 116.78
B_MEAN = 103.94
NORMALIZE = True

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import csv

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def LoadLabelMap(labelmap_path, dict_path):
  """Load index->mid and mid->display name maps.
  Args:
    labelmap_path: path to the file with the list of mids, describing
        predictions.
    dict_path: path to the dict.csv that translates from mids to display names.
  Returns:
    labelmap: an index to mid list
    label_dict: mid to display name dictionary
  """
  labelmap = open(labelmap_path).read().split()

  label_dict = {}
  with open(dict_path, 'r') as f:
	  reader = f.readlines()
	  for line in reader:
	    words = [word.strip(' "\n') for word in line.split(',', 1)]
	    label_dict[words[0]] = words[1]

  return labelmap, label_dict


def predict(image_filename, flag_top_k=20, flags_score_threshold=0.05):

  labelmap = 'classes-trainable.txt'
  cdict = 'class-descriptions.csv'
  model = torch.load('resnet.pth')
  for i, (name, module) in enumerate(model._modules.items()):
      module = recursion_change_bn(model)
  model.eval()
  labelmap, label_dict = LoadLabelMap(labelmap, cdict)
  new_img = np.asarray(Image.open(image_filename).resize((229,229))).transpose((2, 0, 1))

  # print(new_img.shape)
  
  if NORMALIZE:
  	final_img = np.zeros((3, 229,229))
  	R, G, B = new_img
  	R = R - R_MEAN
  	G = G - G_MEAN
  	B = B - B_MEAN
  	final_img[0, :,:], final_img[1,:,:], final_img[2,:,:] = R, G, B
  	# final_img = final_img.transpose((0,0,2))
  else:
  	final_img = new_img
  vals = model(torch.FloatTensor(final_img).unsqueeze(0)).data.numpy()
  vals_sigmoid = F.sigmoid(torch.FloatTensor(vals))

  top_k = vals.argsort()[::-1]
  if flag_top_k > 0:
    top_k = top_k[:flag_top_k]
  if flags_score_threshold is not None:
    top_k = [i for i in top_k
             if vals_sigmoid[i] >= flags_score_threshold]
  for idx in top_k:
    mid = labelmap[idx]
    display_name = label_dict[mid]
    score = vals_sigmoid[idx]
    print('{:04d}: {} - {} (score = {:.2f})'.format(
        idx, mid, display_name, score))















