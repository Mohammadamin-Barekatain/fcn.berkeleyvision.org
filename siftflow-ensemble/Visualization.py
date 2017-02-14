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
import matplotlib.pyplot as plt
import scipy.io as sio

target = "test" #test: test set, trainval: training set
ensemble = True
#models = ['fcn8s-distilled','fcnResNet152-skip']
models =['fcnResNet152','fcn16s','fcn32s','fcn8s','fcnResNet101','fcnResNet50']
cmap = plt.get_cmap('hsv')

n_models = len(models) * 1.0
GPU = int(sys.argv[1])

def comp_score(solver, dataset, model, layer='score'):
    solver.test_nets[0].share_with(solver.net)
    net = solver.test_nets[0]
    n_cl = net.blobs[layer].channels
    for ind, name in enumerate(dataset):
        print datetime.now(), model, "image %d/%d"%(ind,len(dataset))
        net.forward()
        score = net.blobs[layer].data[0]
        im = score.argmax(0)
        plt.imsave(os.path.join('predictions/', layer + '-' + name + '-' + model + '.png'), im, vmin=0, vmax=32, cmap=cmap)
        if model == models[0]:
            im = net.blobs[layer[-3:]].data[0, 0]
            plt.imsave(os.path.join('predictions/', layer + '-' + name + '-gt' + '.png'), im,  vmin=0, vmax=32, cmap=cmap)
        if ensemble:
            score = (score/n_models) + sio.loadmat(layer+'/'+name+'.mat')['S']
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

if ensemble:
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
        
    comp_score(solver, test, model, layer='score_sem')
    comp_score(solver, test, model, layer='score_geo')
       
if ensemble:
    for ind, name in enumerate(test):
        for layer in ['score_sem', 'score_geo']:
            score = sio.loadmat(layer+'/'+name+'.mat')['S']
            im = score.argmax(0)
            plt.imsave(os.path.join('predictions/', layer + '-' + name + '-ensemble' + '.png'), im, vmin=0, vmax=32, cmap=cmap)
