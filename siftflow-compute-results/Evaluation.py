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
import six.moves.cPickle as pickle

GPU = int(sys.argv[1]) # Set GPU Device 
model = 'fcnResNet152-skip' # Set Model Name, should be same as folder name
weight_folder = 'log1' # Set The Snapshot Folder
begin = 1

TST_TAG_LIST = ["overall accuracy", "mean accuracy", "mean IU", "fwavacc", "per-class IU", 'loss']
model_path = '../siftflow-' + model+ '/'+ weight_folder + '/'

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_hist(net, save_dir, dataset, layer='score', gt='label'):
    n_cl = net.blobs[layer].channels
    if save_dir:
        os.mkdir(save_dir)
    hist = np.zeros((n_cl, n_cl))
    loss = 0
    for idx in dataset:
        net.forward()
        hist += fast_hist(net.blobs[gt].data[0, 0].flatten(),
                                net.blobs[layer].data[0].argmax(0).flatten(),
                                n_cl)

        if save_dir:
            im = Image.fromarray(net.blobs[layer].data[0].argmax(0).astype(np.uint8), mode='P')
            im.save(os.path.join(save_dir, idx + '.png'))
        # compute the loss as well
        loss += net.blobs['loss'].data.flat[0]
    return hist, loss / len(dataset)

def seg_tests(solver, save_format, dataset, layer='score', gt='label', iter=1):
    print '>>>', datetime.now(), 'Begin seg tests', iter
    solver.test_nets[0].share_with(solver.net)
    return do_seg_tests(solver.test_nets[0], iter, save_format, dataset, layer, gt)

def do_seg_tests(net, iter, save_format, dataset, layer='score', gt='label'):
    ret = {x:-1 for x in TST_TAG_LIST}
    ret['Iteration'] = iter
    n_cl = net.blobs[layer].channels
    if save_format:
        save_format = save_format.format(iter)
    hist, loss = compute_hist(net, save_format, dataset, layer, gt)
    # mean loss
    #print '>>>', datetime.now(), 'Iteration', iter, 'loss', loss
    ret['loss'] = loss
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    #print '>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
    ret['overall accuracy'] = acc
    # per-class accuracy
    acc = np.diag(hist) / hist.sum(1)
    #print '>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
    ret['mean accuracy'] = np.nanmean(acc)
    # per-class IU
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    #print '>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
    ret['mean IU'] = np.nanmean(iu)
    #IU per class
    #print 'Iteration', iter, 'per-class IU', iu
    ret['per-class IU'] = iu
    freq = hist.sum(1) / hist.sum()
    #print '>>>', datetime.now(), 'Iteration', iter, 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum()
    ret['fwavacc'] = (freq[freq > 0] * iu[freq > 0]).sum()
    return ret


def setup(model):
    with open('solver.prototxt') as f:
        with open('temp','w') as w:
            w.write('train_net: \''+model+'-trainval.prototxt\'\n')
            w.write('test_net: \'' + model + '-test.prototxt\'\n')
            flag = True
            for line in f:
                if flag:
                    next(f)
                    flag = False
                    continue
                w.write(line)

    os.system('mv temp solver.prototxt')

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass
setup(model)

# init
caffe.set_device(GPU)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)
test = np.loadtxt('../data/sift-flow/test.txt', dtype=str)
error_sem = []
error_geo = []
# scoring
for i in xrange(begin,500):
    iter = i*2500
    weights = model_path + weight_folder+'_iter_'+str(iter)+'.caffemodel'
    if not os.path.isfile(weights):
        print 'Used',(i-1),'files'
        break
    solver.net.copy_from(weights)
    error_sem.append(seg_tests(solver, False, test, layer='score_sem', gt='sem', iter=iter))
    error_geo.append(seg_tests(solver, False, test, layer='score_geo', gt='geo', iter=iter))

with open(model + '-' + weight_folder + '-sem.pkl', 'wb') as f:
    pickle.dump(error_sem, f)
with open(model + '-' + weight_folder + '-geo.pkl', 'wb') as f:
    pickle.dump(error_geo, f)


