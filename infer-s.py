import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
name = "opencountry_land359"
im = Image.open('data/sift-flow/Images/spatial_envelope_256x256_static_8outdoorcategories/'+name+'.jpg')
in_ = np.array(im, dtype=np.float32)
plt.imsave("im", in_)
in_ = in_[:,:,::-1]
in_ -= np.array((114.578, 115.294, 108.353))
in_ = in_.transpose((2,0,1))
print "image loaded"
# load net
net = caffe.Net('siftflow-fcn16s/deploy.prototxt', 'siftflow-fcn16s/siftflow-fcn16s-heavy.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take a	rgmax for prediction
net.forward()
out = net.blobs['score_sem'].data[0].argmax(axis=0)
print out
plt.imsave(name, out)
