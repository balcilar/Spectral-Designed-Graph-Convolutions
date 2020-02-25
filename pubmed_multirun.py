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
import os



# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 250, 'Number of epochs to train.')
flags.DEFINE_list('nfilter', [16,3], 'Number of units in each hidden layer for instance [160, 100,50]')
flags.DEFINE_list('activation_funcs', [tf.nn.relu,tf.nn.relu], 'Activation functions for hidden+output layers  [tf.nn.relu, lambda x: x]')
flags.DEFINE_list('biases', [False,False], 'if apply bias for hidden and output layers')
flags.DEFINE_list('isdroput_inp', [True,True], 'if apply dropout for hidden and output layers'' input')
flags.DEFINE_list('isdroput_kernel', [False,False], 'if apply dropout for hidden and output layers'' kernel')
flags.DEFINE_list('firstDWS_learnable', [True,True], 'if first kernel''s depthwise weights are learnable or not')
flags.DEFINE_list('isdepthwise', [True,True], 'if layer is depthwise or not')
flags.DEFINE_float('dropout', 0.25, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.') 
flags.DEFINE_float('weight_decay_depthwise', 5e-3, 'Weight for L2 loss on depthwise weigths.')





# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
orgtrain_mask=train_mask.copy()

# Some preprocessing
#features = preprocess_features(features)
features = justpreprocess_features(features)
feat=features.toarray()

rfile='logs/pubmedresult.txt'
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

    if os.path.exists('pubmedA0.mat') and  os.path.exists('pubmedA1.mat'):    
            
        aa=sio.loadmat('pubmedA0.mat')    
        A0=aa['A0']       
           
        aa=sio.loadmat('pubmedA1.mat')
        A1=aa['A1']  
        
    else:
        print(' Eigenvectors Calculation')
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
        print('Creating low pass kernel')
        dbb=(V1.max()-V1)/V1.max()
        db=dbb**3 
        A0=U1.dot(np.diag(db).dot(U1.T))
        sio.savemat('pubmedA0.mat',{'A0':A0})

        # high pass filter
        print('Creating high pass kernel')
        dbb=np.linspace(0,1,db.shape[0])         
        A1=U1.dot(np.diag(dbb).dot(U1.T))
        sio.savemat('pubmedA1.mat',{'A1':A1})

    support2 = list()
    support2.append(A0) 
    support2.append(A1)  
    
    num_supports = len(support2)

else:  
    exit


stop=0

cleangcnn=[]
cleanxentgcnn=[]
semisuper=[];semisuperx=[]
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
        'num_features_nonzero': tf.placeholder(tf.int32)        
    }

    model = DSSGCN(placeholders2, input_dim=feat.shape[1], logging=True)  

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    cost_val = []
    bvalacc=0
    bvalcost=10000
    btestcost=0
    btestacc=0

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

       
        
        # if validation entropy is better, keep the model
        if bvalcost>valx:
            bvalcost=valx
            bestpredict=predv.copy()
            model.save(sess)

        # # Print results
        f=open(rfile,'a')
        msg=str(iter)+ ", Epoch:,"+ '%04d' % (epoch + 1) + ", train_xent=,"+ "{:.5f}".format(outs[1])+", train_acc=,"+ "{:.5f}".format(outs[2]) + ", val_xent=,"+ "{:.5f}".format(valx)+", val_acc=,"+ "{:.5f}".format(acc)+", test_acc=,"+ "{:.5f}".format(testacc)+'\n'
        
        f.writelines(msg)
        f.close()
        if True:
          print(iter, ", Epoch:,", '%04d' % (epoch + 1), ", train_xent=,", "{:.5f}".format(outs[1]),", train_acc=,", "{:.5f}".format(outs[2]), 
          ", val_xent=,", "{:.5f}".format(valx),", val_acc=,", "{:.5f}".format(acc),", test_acc=,", "{:.5f}".format(testacc))
     

    print("Optimization Finished! best model is loading")

    model.load(sess)

    # Testing
    cost, acc, valx,predv = evaluate(feat, support2, y_val, val_mask, placeholders2)
    tcost, tacc, tx,predt = evaluate(feat, support2, y_test, test_mask, placeholders2)

    btestacc=tacc #(np.argmax(y_test[test_mask],axis=1)==predv[test_mask]).mean()

    print("Test set results:", "accuracy=", "{:.5f}".format(btestacc), " entropy=", "{:.5f}".format(tx))
    cleangcnn.append(btestacc)
    cleanxentgcnn.append(tx)
    

    del model
    sess.close()

    tf.keras.backend.clear_session()

    print(iter, "  ",np.array(cleangcnn).mean(),"  ",np.array(cleangcnn).std())
    
    