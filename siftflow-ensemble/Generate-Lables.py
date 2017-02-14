# coding: utf-8
from __future__ import division
import string
import os
import numpy as np
import caffe
import surgery
import os
import sys
import caffe
from datetime import datetime
from PIL import Image
from operator import itemgetter, attrgetter
import scipy.io as sio

GPU = 0 # Set GPU Device
target = "test"
models = ['fcn8s','fcn16s', 'fcn32s' ,'fcnResNet152', 'fcnResNet101','fcnResNet50'] # Set Models Name, should be same as folder name
n_models = len(models) * 1.0

def comp_score(solver, dataset, layer='score'):
    solver.test_nets[0].share_with(solver.net)
    net = solver.test_nets[0]
    n_cl = net.blobs[layer].channels
    for ind, name in enumerate(dataset):
        print datetime.now(), "image %d/%d"%(ind,len(dataset))
        net.forward()
        score = (net.blobs[layer].data[0]/n_models) + sio.loadmat(layer+'/'+name+'.mat')['S']
        sio.savemat(layer+'/'+name+'.mat', {'S':score})


def setup(model):
    with open('solver.prototxt') as f:
        with open('temp','w') as w:
            w.write('train_net: \''+model+'-trainval.prototxt\'\n')
            w.write('test_net: \'' +model+ '-'+target+'.prototxt\'\n')
            flag = True
            for line in f:
                if flag:
                    next(f)
                    flag = False
                    continue
                w.write(line)

    os.system('mv temp solver.prototxt')
                

test = np.loadtxt('../data/sift-flow/'+target+'.txt', dtype=str)

for name in test:
    sio.savemat('score_sem/' +name+ '.mat', {'S':np.zeros((33, 256, 256))})
    sio.savemat('score_geo/' +name+ '.mat', {'S':np.zeros((3, 256, 256))})

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

# init
caffe.set_device(GPU)
caffe.set_mode_gpu()
# scoring

    
for model in models:
    print datetime.now(), "Running model: ", model
    model_path = '../siftflow-' + model + '/'
    setup(model)
    weights = model_path + '_iter_2500'+'.caffemodel'
    solver = caffe.SGDSolver('solver.prototxt')
    solver.net.copy_from(weights)
    # surgeries
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    surgery.interp(solver.net, interp_layers)
        
    comp_score(solver, test, layer='score_sem')
    comp_score(solver, test, layer='score_geo')
       

