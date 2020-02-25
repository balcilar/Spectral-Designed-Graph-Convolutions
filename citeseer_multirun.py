from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import DSSGCN
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy
from scipy.sparse import csr_matrix, lil_matrix



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_list('nfilter', [160,6], 'Number of units in each layer for instance [160, 100,50]')
flags.DEFINE_list('activation_funcs', [tf.nn.relu,lambda x: x], 'Activation functions for hidden+output layers  [tf.nn.relu, lambda x: x]')
flags.DEFINE_list('biases', [False,True], 'if apply bias for hidden and output layers')
flags.DEFINE_list('isdroput_inp', [True,True], 'if apply dropout for hidden and output layers'' input')
flags.DEFINE_list('isdroput_kernel', [True,True], 'if apply dropout for hidden and output layers'' kernel')
flags.DEFINE_list('firstDWS_learnable', [True,True], 'if first kernel''s depthwise weights are learnable or not')
flags.DEFINE_list('isdepthwise', [True,True], 'if layer is depthwise or not')
flags.DEFINE_float('dropout', 0.75, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 3e-4, 'Weight for L2 loss on embedding matrix.') 
flags.DEFINE_float('weight_decay_depthwise', 3e-3, 'Weight for L2 loss on depthwise weigths.')
flags.DEFINE_integer('early_stopping', 400, 'Tolerance for early stopping (# of epochs).')


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask,label = load_data(FLAGS.dataset,fullabel=True)
orgtrain_mask=train_mask.copy()

# train_mask=~(val_mask + test_mask)
# I=train_mask==True
# y_train[I,:]=label[I,:]


# Some preprocessing
#features = preprocess_features(features)
features = justpreprocess_features(features)
feat=features.toarray()


rfile='logs/citeseer.txt'
f=open(rfile,'w')

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy,model.entropy,model.outputs], feed_dict=feed_dict_val)
    res=np.argmax(outs_val[3],axis=1)
    return outs_val[0], outs_val[1], outs_val[2],res

# cleaning

if FLAGS.dataset=='pubmed':
    exit
else:   

    
    W=1.0*adj.toarray()
    d = W.sum(axis=0)
    

    # normalized Laplacian matrix.
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=np.eye(D.shape[0])-(W.dot(D)).T.dot(D)
    V1,U1 = np.linalg.eigh(nL) 
    V1[V1<0]=0 

    # low pass filter
    dbb=(V1.max()-V1)/V1.max()
    db=dbb**5 

       
    support2 = list()
    A0=U1.dot(np.diag(db).dot(U1.T))
    A0[np.where(np.abs(A0)<0.001)]=0    
    support2.append(A0)

    # all pass filters
    support2.append(np.eye(A0.shape[0]))
    # band pass filters
    # ff=np.linspace(0,V1.max(),5)  
    # for f in ff[1:-1]:
    #     db4=np.exp(-(((V1-f)*1)**2))          
    #     A2=U1.dot(np.diag(db4).dot(U1.T))
    #     A2[np.where(np.abs(A2)<0.001)]=0    
    #     support2.append(A2) 

    num_supports = len(support2)


cleangcnn=[]
cleanxentgcnn=[]
semisuper=[];semisuperx=[];semisuper2=[]
for iter in range(0,20):
    # Set random seed
    seed = iter
    np.random.seed(seed)
    tf.set_random_seed(seed)


    placeholders2 = {        
        'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],    
        'features': tf.placeholder(tf.float32, shape=(None, feat.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    model = DSSGCN(placeholders2, input_dim=feat.shape[1], logging=True)  

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    cost_val = []
    bvalacc=0
    bvalcost=10000
    btestcost=0
    btestacc=0
    besttrain=0
    early=0
    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(feat, support2, y_train, train_mask, placeholders2)
        feed_dict.update({placeholders2['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.entropy, model.accuracy], feed_dict=feed_dict)

        # Validation
        cost, acc, valx,predv = evaluate(feat, support2, y_val, val_mask, placeholders2)
        cost_val.append(cost)

        testacc=(np.argmax(y_test[test_mask],axis=1)==predv[test_mask]).mean()


        if bvalacc<acc:  #bvalcost>valx:
            
            semipredt=predv.copy()
            bvalacc=acc
            btestacc=testacc
            model.save(sess)
            

        if bvalcost>valx:
            bvalcost=valx
            #model.save(sess)

        
        f=open(rfile,'a')
        msg=str(iter)+ ", Epoch:,"+ '%04d' % (epoch + 1) + ", train_xent=,"+ "{:.5f}".format(outs[1])+", train_acc=,"+ "{:.5f}".format(outs[2]) + ", val_xent=,"+ "{:.5f}".format(valx)+", val_acc=,"+ "{:.5f}".format(acc)+", test_acc=,"+ "{:.5f}".format(testacc)+'\n'
        
        f.writelines(msg)
        f.close()

        # # Print results
        if True:
          print("Epoch:", '%04d' % (epoch + 1), "train_xent=", "{:.5f}".format(outs[1]),
            "train_acc=", "{:.5f}".format(outs[2]), "val_xent=", "{:.5f}".format(valx),
            "val_acc=", "{:.5f}".format(acc), "besttest_acc=", "{:.5f}".format(btestacc),"curr test=", "{:.5f}".format(testacc))
        if early>=FLAGS.early_stopping:
            print("Early stopping...")
            break
        

    cleangcnn.append(btestacc)
    #cleanxentgcnn.append(0)


    del model
    sess.close()

    tf.keras.backend.clear_session()

    print(iter, "          Test acc: ",np.array(cleangcnn).mean(),"  std:",np.array(cleangcnn).std())
    