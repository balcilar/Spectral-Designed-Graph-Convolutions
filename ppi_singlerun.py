from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import DSSGCN_NOBATCH
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import f1_score
import numpy as np
import os

# Settings  
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_list('nfilter', [800,800,800,800,800,800,121], 'Number of units in each hidden layers for instance [160, 100,50]')
flags.DEFINE_list('activation_funcs', [tf.nn.relu,tf.nn.relu,tf.nn.relu,tf.nn.relu,tf.nn.relu,tf.nn.relu,lambda x: x], 'Activation functions for hidden+output layers  [tf.nn.relu, lambda x: x]')
flags.DEFINE_list('biases', [True,True,True,True,True,True,True], 'if apply bias for hidden and output layers')
flags.DEFINE_list('isdroput_inp', [False,False,False,False,False,False,False], 'if apply dropout for hidden and output layers'' input')
flags.DEFINE_list('isdroput_kernel', [False,False,False,False,False,False,False], 'if apply dropout for hidden and output layers'' kernel')
flags.DEFINE_list('firstDWS_learnable', [True,True,True,True,True,True,True], 'if first kernel''s depthwise weights are learnable or not')
flags.DEFINE_list('isdepthwise', [True,True,True,True,True,True,True] , 'if layer is depthwise or not')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'Weight for L2 loss on embedding matrix.') 
flags.DEFINE_float('weight_decay_depthwise', 0.0, 'Weight for L2 loss on depthwise weigths.')
flags.DEFINE_integer('nkernel', 3,'number of kernels')
flags.DEFINE_string('kerneltype', 'custom', 'type of kernel cheb, gcn, custom') 

# 

nkernel=flags.FLAGS.nkernel
kerneltype=flags.FLAGS.kerneltype


a=sio.loadmat('data/ppimat.mat')

# list of adjacency matrix
A=a['A'][0]
# list of features
F=a['F'][0]
# list of output
Y=a['Y'][0]


if not os.path.exists('ppiUVC.mat'):
    print(' Eigenvectors Calculation')
    U=[];V=[]
    for i in range(0,24):
        W=1.0*A[i]
        d = W.sum(axis=0)
        # # normalized Laplacian matrix.
        # dis=1/np.sqrt(d)
        # dis[np.isinf(dis)]=0
        # dis[np.isnan(dis)]=0
        # D=np.diag(dis)
        # nL=np.eye(D.shape[0])-(W.dot(D)).T.dot(D)
        nL=np.diag(d)-W
        V1,U1 = np.linalg.eigh(nL) 
        V1[V1<0]=0    
        U.append(U1)
        V.append(V1)
    sio.savemat('ppiUVC.mat',{'U':U,'V':V})
    print(' Eigenvectors saved')
else:
    a=sio.loadmat('ppiUVC.mat')
    # list of eigenvalues
    V=a['V'][0]
    # list of eigenvectors
    U=a['U'][0]
    for i in range(0,24):
        V[i]=V[i][0]


vmax=0
for v in V:
    vmax=max(vmax,v.max())

A0=[];A1=[];A2=[];SP=[]

# prepare convolution kernels
for i in range(0,24):
    A0=[]
    #Y[i]=Y[i][:,0:1]
    V[i][V[i]<0]=0

    if kerneltype=='cheb':
        chebnet = chebyshev_polynomials(A[i], nkernel-1,True)
        for j in range(0,nkernel):
            A0.append(chebnet[j].toarray())
        SP.append(A0)
    elif kerneltype=='gcn':
        A0=[(normalize_adj(A[i] + sp.eye(A[i].shape[0]))).toarray()]
        SP.append(A0)
    else:        
        v=2*V[i]/vmax

        # low pass conv
        dv=10    
        db=np.exp(-(V[i])/dv)    
        a0=U[i].dot(np.diag(db).dot(U[i].T))
        a0[np.where(np.abs(a0)<0.001)]=0 
        A0.append(a0)

        # high pass conv
        db1=np.linspace(0,1,db.shape[0])**(1/1)  
        #db1=V[i]/V[i].max() 
        a1=U[i].dot(np.diag(db1).dot(U[i].T))
        a1[np.where(np.abs(a1)<0.001)]=0    
        A0.append(a1)

        # all pass convulation
        A0.append(np.eye(a1.shape[0]))


        SP.append(A0)

# define which kernel's will be used in the model 0:lowpass, 1:highpass 2:all pass
usedkernel=[0,1,2]
num_supports=len(usedkernel)

# set your seed number 
seed = 0 
np.random.seed(seed)
tf.set_random_seed(seed)   



placeholders = {        
        'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],    
        'features': tf.placeholder(tf.float32, shape=(None, F[0].shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, Y[0].shape[1])),        
        'dropout': tf.placeholder_with_default(0., shape=()),
        #'istrain': tf.placeholder(tf.bool),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
        
    }

model = DSSGCN_NOBATCH(placeholders, input_dim=50, logging=True)  

sess = tf.Session()
sess.run(tf.global_variables_initializer())

bval=0
best=0
for epoch in range(FLAGS.epochs):
    p=np.random.permutation(20)
    # Training step
    n=0
    tracc=0
    for idd in range(0,20):

        id=p[idd]
        # Construct feed dictionary
        spp=[SP[id][i] for i in usedkernel]  #[A0[id],A1[id]] #sp=[A0[id],A1[id],A2[id]]

        feed_dict = construct_feed_dict_inductive(F[id], spp, Y[id], placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})  
        #feed_dict.update({placeholders['istrain']: True})      
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        n+=F[id].shape[0]
        tracc+=F[id].shape[0]*outs[2]
    tracc/=n

    # print(epoch,"  ",tracc)
    # continue

    # Validation
    n=0
    tvacc=0
    for id in range(20,22): #range(20,22):
        spp=[SP[id][i] for i in usedkernel] #[A0[id], A1[id]] #sp=[A0[id],A1[id],A2[id]]
        feed_dict = construct_feed_dict_inductive(F[id], spp, Y[id], placeholders)
        #feed_dict.update({placeholders['istrain']: False})
        outs_val = sess.run([model.loss, model.accuracy, model.entropy,model.outputs], feed_dict=feed_dict)        
        n+=F[id].shape[0]
        tvacc+=F[id].shape[0]*outs_val[1]
    tvacc/=n

    # test
    n=0
    tsacc=0
    pr=np.zeros((0,121))
    gt=np.zeros((0,121))
    for id in range(22,24): #range(22,24):
        spp=[SP[id][i] for i in usedkernel] #[A0[id],A1[id]] #sp=[A0[id],A1[id],A2[id]]
        feed_dict = construct_feed_dict_inductive(F[id], spp, Y[id], placeholders)
        #feed_dict.update({placeholders['istrain']: False})
        outs_test = sess.run([model.loss, model.accuracy, model.entropy,model.outputs], feed_dict=feed_dict)   
        pred=(np.sign(outs_test[3])+1)/2
        pr=np.vstack((pr,pred))
        gt=np.vstack((gt,Y[id]))             
        n+=F[id].shape[0]
        tsacc+=F[id].shape[0]*outs_test[1]
    try:
        f1=f1_score(gt, pr, average='micro')
    except:
        f1=0

    tsacc/=n
    if bval<tvacc:
        best=f1
        bval=tvacc
    print(epoch,"  train_acc:",tracc,"   val_acc",tvacc,"  test_acc",tsacc,"  test_f1",f1,"  best_f1_wrt_val:",best)
