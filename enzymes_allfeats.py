from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import DSSGCN_GC_BATCH
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd




# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_list('hidden', [200,200,'meanmax',-100,-6], 'Number of units in each layer negative:denselayer, positive:ConvGraph layer ')
flags.DEFINE_list('activation_funcs', [tf.nn.relu,tf.nn.relu, None, tf.nn.relu,lambda x: x], 'Activation functions for layers  [tf.nn.relu, lambda x: x]')
flags.DEFINE_list('biases', [False,False,None,True,True], 'if apply bias on layers')
flags.DEFINE_list('isdroput_inp', [True,True,None,True,True], 'if apply dropout on layers'' input')
flags.DEFINE_list('isdroput_kernel', [True,True,None,False,False], 'if apply dropout on layers'' kernel')
flags.DEFINE_list('firstDWS_learnable', [True,True,None,None,None], 'if first kernel''s depthwise weights are learnable or not')
flags.DEFINE_list('isdepthwise', [False,False,None,None,None], 'if layer is depthwise or not')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.') 
flags.DEFINE_float('weight_decay_depthwise', 1e-4, 'Weight for L2 loss on depthwise weigths.')
flags.DEFINE_integer('nkernel', 4, 'number of kernels') #4
flags.DEFINE_integer('numbatch', 3, 'number of update in each epoch')
flags.DEFINE_string('kerneltype', 'custom', 'type of kernel cheb, gcn, custom')   #custom







bsize=flags.FLAGS.numbatch
nkernel=flags.FLAGS.nkernel
kerneltype=flags.FLAGS.kerneltype

a=sio.loadmat('data/enzymes.mat')
# list of adjacency matrix
A=a['A'][0]
# list of features
F=a['F'][0]
Y=a['Y'][0]




U=[];V=[]
dmax=0
dmin=1000
x=1e-5 #0.1 #0.00001
for i in range(0,len(A)):
    W=1.0*A[i]  # + x - x*np.eye(A[i].shape[0])
    d = W.sum(axis=0)
    dmax=max(dmax,d.max())
    dmin=min(dmin,d.min())
    # Laplacian matrix. 
    # D = np.diag(d)
    # L = D - W
    # V1,U1 = np.linalg.eigh(L) 
    # V1[V1<0]=0
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=np.eye(D.shape[0])-(W.dot(D)).T.dot(D)
    V1,U1 = np.linalg.eigh(nL) 
    V1[V1<0]=0
    U.append(U1)
    V.append(V1)




vmax=0
nmax=0
for v in V:    
    vmax=max(vmax,v.max())
    nmax=max(nmax,v.shape[0])

globalmax=vmax

A0=[];A1=[];A2=[]
ND=np.zeros((len(A),1)) 
FF=np.zeros((len(A),nmax,21+1)) #3  21
YY=np.zeros((len(A),6))
SP=np.zeros((len(A),nkernel,nmax,nmax))



# prepare convolution supports

for i in range(0,len(A)):  
    n=F[i].shape[0]    
    FF[i,0:n,0:21]= F[i]#[:,0:3]  
    # add node degree as feature
    FF[i,0:n,-1]= A[i].sum(0) 
    
    ND[i,0]=n
    YY[i,Y[i]]=1

    dv=10

    vmax= V[i].max() 

    if kerneltype=='cheb':
        chebnet = chebyshev_polynomials(A[i], nkernel-1,True)
        for j in range(0,nkernel):
            SP[i,j,0:n,0:n]=chebnet[j].toarray()
    elif kerneltype=='gcn':
        SP[i,0,0:n,0:n]= (normalize_adj(A[i] + sp.eye(A[i].shape[0]))).toarray()
    else:        
        SP[i,0,0:n,0:n]=np.eye(n)  
        SP[i,1,0:n,0:n]=U[i].dot(np.diag(np.exp(-(1*(V[i]-0.0)**2))).dot(U[i].T))
        SP[i,2,0:n,0:n]=U[i].dot(np.diag(np.exp(-(1*(V[i]-vmax*0.5)**2))).dot(U[i].T))
        SP[i,3,0:n,0:n]=U[i].dot(np.diag(np.exp(-(1*(V[i]-vmax)**2))).dot(U[i].T))   
        

    
num_supports=SP.shape[1]


def normalize_wrt_train(FF,ND,trid):
    tmp=np.zeros((0,FF[0].shape[1]))
    for i in trid:
        tmp=np.vstack((tmp,FF[i][0:int(ND[i]),:]))
    avg=tmp.mean(0)
    st=tmp.std(0)
    FFF=FF.copy()
    for i in range(0,len(FFF)):
        tmp2=(FFF[i][0:int(ND[i]),:]-avg)/st
        tmp2[:,4]=0
        tmp2[:,3]=0
        FFF[i][0:int(ND[i]),:]=tmp2
    return FFF



for iter in range(0,20):
    seed = iter 
    np.random.seed(seed)
    tf.set_random_seed(seed)

    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
    idx_list = []
    for idx in skf.split(np.zeros(len(Y)), Y):
        idx_list.append(idx) 

    tprediction=[]
    TS=[]
    NB=np.zeros((FLAGS.epochs,10))




    for fold in range(0,10):

        trid, tsid = idx_list[fold]  
        
        FFF=normalize_wrt_train(FF,ND,trid)
        
        

        placeholders = {        
                'support': tf.placeholder(tf.float32, shape=(None,num_supports,nmax,nmax)),
                'features': tf.placeholder(tf.float32, shape=(None,nmax, FFF.shape[2])),
                'labels': tf.placeholder(tf.float32, shape=(None, 6)),  
                'nnodes': tf.placeholder(tf.float32, shape=(None, 1)),               
                'dropout': tf.placeholder_with_default(0., shape=())       
            }

        model = DSSGCN_GC_BATCH(placeholders, input_dim=FFF.shape[2], logging=True)  

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        

        feed_dictT = dict()
        feed_dictT.update({placeholders['labels']: YY[tsid,:]})    
        feed_dictT.update({placeholders['features']: FFF[tsid,:,:]})
        feed_dictT.update({placeholders['support']: SP[tsid,:,:,:]})    
        feed_dictT.update({placeholders['nnodes']: ND[tsid,]})     
        feed_dictT.update({placeholders['dropout']: 0})

        
        ytest=YY[tsid,:] 
        ind=np.round(np.linspace(0,len(trid),bsize+1))

        
        besttr=100
        for epoch in range(FLAGS.epochs):
            #otrid=trid.copy()
            np.random.shuffle(trid)
            ent=[]
            for i in range(0,bsize):
                feed_dictB = dict()
                bid=trid[int(ind[i]):int(ind[i+1])]
                feed_dictB.update({placeholders['labels']: YY[bid,:]})    
                feed_dictB.update({placeholders['features']: FFF[bid,:,:]})
                feed_dictB.update({placeholders['support']: SP[bid,:,:,:]})             
                feed_dictB.update({placeholders['nnodes']: ND[bid,]})            
                feed_dictB.update({placeholders['dropout']: FLAGS.dropout})
                outs = sess.run([model.opt_op,model.entropy], feed_dict=feed_dictB)
                ent.append(outs[1])
            
            outsT = sess.run([model.accuracy, model.loss, model.entropy,model.outputs], feed_dict=feed_dictT)
            
            vtest=np.sum(np.argmax(outsT[3],1)==np.argmax(ytest,1))            

            NB[epoch,fold]=vtest
            if np.mod(epoch + 1,1)==0 or epoch==0:
                print(fold," Epoch:", '%04d' % (epoch + 1),"train_xent=", "{:.5f}".format(np.mean(ent)), "test_loss=", "{:.5f}".format(outsT[1]), 
                "test_xent=", "{:.5f}".format(outsT[2]), "test_acc=", "{:.5f}".format(outsT[0]), " ntrue=", "{:.0f}".format(vtest))
            a=1
    
    fname='logs/enzyms_fullfeat_'+ kerneltype+'_nk'+str(nkernel)+'_'+str(iter)+'.csv'
    pd.DataFrame(NB).to_csv(fname) 
