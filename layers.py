from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = None #placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        #output=tf.nn.l2_normalize(output,axis=1)
        return self.act(output)





class GraphConvolutionwithDephSep(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=(0.,0.),act=tf.nn.relu, bias=False,firstDSWS=True,
                 featureless=False,isdepthwise=True, **kwargs):
        super(GraphConvolutionwithDephSep, self).__init__(**kwargs)

        self.isdropout=dropout
        if dropout[0] or dropout[1]:
            self.dropout = placeholders['dropout']            
        else:
            self.dropout = 0.

        self.firstDSWS=firstDSWS
        self.act = act
        self.support = placeholders['support']        
        self.featureless = featureless
        self.bias = bias  
        self.isdepthwise=isdepthwise
        #self.istrain = placeholders['istrain'] 


        

        with tf.variable_scope(self.name + '_vars'):
            if self.isdepthwise:
                i=0
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],name='weights_' + str(i))

                if len(self.support)>1:
                    if self.firstDSWS:
                        self.vars['sdweight_' + str(i)] = ones([input_dim],name='sdweight_' + str(i)) 
                        
                    for i in range(1,len(self.support)):
                        self.vars['sdweight_' + str(i)] = zeros([input_dim],name='sdweight_' + str(i)) 
            else:
                for i in range(0,len(self.support)):
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],name='weights_' + str(i))

                    

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.isdropout[0]:
            x = tf.nn.dropout(x, 1-self.dropout)
            

        # convolve

        if self.isdepthwise:

            supports = list()
            
            for i in range(0,len(self.support)):
                if self.isdropout[1]:
                    tmp=tf.nn.dropout(self.support[i], 1-self.dropout) 
                    s0=tf.matmul(tmp, x, a_is_sparse=True)
                else:
                    s0=tf.matmul(self.support[i], x, a_is_sparse=True)

                if len(self.support)>1 and (i>0 or self.firstDSWS):
                    s0=s0*self.vars['sdweight_'+str(i)]
                supports.append(s0) 
            output = tf.add_n(supports)        
            output=tf.matmul(output, self.vars['weights_' + str(0)])
        else:
            supports = list()
            for i in range(0,len(self.support)):
                if self.isdropout[1]:
                    tmp=tf.nn.dropout(self.support[i], 1-self.dropout) 
                    s0=tf.matmul(tmp, x, a_is_sparse=True)
                else:
                    s0=tf.matmul(self.support[i], x, a_is_sparse=True)
                s0=tf.matmul(s0, self.vars['weights_' + str(i)])
                supports.append(s0) 
            output = tf.add_n(supports)               



        # bias
        if self.bias:
            output += self.vars['bias']

        # output = tf.contrib.layers.batch_norm(output, 
        #                                   center=True, scale=True, 
        #                                   is_training=self.istrain)

        return self.act(output)


class GraphConvolutionwithDephSepBatch(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=(0.,0.),act=tf.nn.relu, bias=False,firstDSWS=True,
                 isdepthwise=True,featureless=False, **kwargs):
        super(GraphConvolutionwithDephSepBatch, self).__init__(**kwargs)

        self.isdropout=dropout
        if dropout[0] or dropout[1]:
            self.dropout = placeholders['dropout']            
        else:
            self.dropout = 0.

        self.firstDSWS=firstDSWS
        self.isdepthwise=isdepthwise
        self.act = act
        self.support = placeholders['support']        
        self.featureless = featureless
        self.bias = bias  
        #self.istrain = placeholders['istrain']         

        with tf.variable_scope(self.name + '_vars'):
            i=0
            self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],name='weights_' + str(i))

            if self.support.shape[1]>1 and self.isdepthwise:
                if self.firstDSWS:
                    self.vars['sdweight_' + str(i)] = ones([input_dim],name='sdweight_' + str(i))
                    #self.vars['sdweight_' + str(i)] = glorot([input_dim,1],name='sdweight_' + str(i)) 
                    #self.vars['sdweight_' + str(i)]=tf.squeeze(self.vars['sdweight_' + str(i)])
                      
                for i in range(1,self.support.shape[1]):
                    self.vars['sdweight_' + str(i)] = zeros([input_dim],name='sdweight_' + str(i)) 
                    #self.vars['sdweight_' + str(i)] = glorot([input_dim,1],name='sdweight_' + str(i)) 
                    #self.vars['sdweight_' + str(i)]=tf.squeeze(self.vars['sdweight_' + str(i)])

            if not self.isdepthwise:
                for i in range(1,self.support.shape[1]):
                    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],name='weights_' + str(i))

                    

            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.isdropout[0]:
            x = tf.nn.dropout(x, 1-self.dropout)
            

        # convolve

        if self.isdepthwise:

            supports = list()
            
            for i in range(0,self.support.shape[1]):
                if self.isdropout[1]:
                    tmp=tf.nn.dropout(self.support[:,i,:,:], 1-self.dropout) 
                    s0=tf.matmul(tmp,x) 
                else:
                    s0=tf.matmul(self.support[:,i,:,:],x)

                if self.support.shape[1]>1 and (i>0 or self.firstDSWS):
                    s0=s0*self.vars['sdweight_'+str(i)]  
                supports.append(s0)  

            output = tf.add_n(supports)        
            output=tf.tensordot(output,self.vars['weights_' + str(0)],[2, 0]) 

        else:
            supports = list()
            for i in range(0,self.support.shape[1]):
                if self.isdropout[1]:
                    tmp=tf.nn.dropout(self.support[:,i,:,:], 1-self.dropout) 
                    s0=tf.matmul(tmp,x) 
                else:
                    s0=tf.matmul(self.support[:,i,:,:],x)

                s0=tf.tensordot(s0,self.vars['weights_' + str(i)],[2, 0])
                supports.append(s0)
            output = tf.add_n(supports)
            

        # bias
        if self.bias:
            output += self.vars['bias']

        # output = tf.contrib.layers.batch_norm(output, 
        #                                   center=True, scale=True, 
        #                                   is_training=self.istrain)

        return self.act(output)



class AggLayer(Layer):
    """Graph convolution layer."""
    def __init__(self, placeholders,method='mean',**kwargs):
        super(AggLayer, self).__init__(**kwargs)
        self.ND=placeholders['nnodes']          

        self.method=method

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        if self.method=='mean':
            output=tf.reduce_sum(x,1)/self.ND   
        elif self.method=='max':
            output=tf.reduce_max(x,1)
        else:
            output=tf.concat([tf.reduce_sum(x,1)/self.ND, tf.reduce_max(x,1)], 1)
            
        return output

class PoolLayer(Layer):
    """Graph convolution layer."""
    def __init__(self, placeholders,method='mean',**kwargs):
        super(PoolLayer, self).__init__(**kwargs)
        #self.ND=placeholders['nnodes']          
        self.adj = placeholders['adj'] 
        

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # a=tf.tile(tf.expand_dims(x[:,:,0],-1),[1,1,x.shape[1]])
        # tmp=tf.expand_dims(tf.reduce_max(self.adj*a,2),-1)

        # for i in range(1,x.shape[2]):
        #     a=tf.tile(tf.expand_dims(x[:,:,i],-1),[1,1,x.shape[1]])
        #     tmp=tf.concat([tmp,tf.expand_dims(tf.reduce_max(self.adj*a,2),-1)],2)
        # output=tmp

        output=tf.matmul(self.adj,x)    #tf.concat([tf.expand_dims(tmp,-1),tmp2],2).shape    
        
        return output

