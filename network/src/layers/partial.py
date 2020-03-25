from __future__ import print_function

from keras import activations, initializers, constraints
from keras import regularizers
from keras.engine import Layer
import keras.backend as K
import scipy.sparse as sp
import numpy as np
import tensorflow as tf


class PartialConnection_MF(Layer):

    def __init__(self, adjlist, 
                 share_weights=True,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(PartialConnection_MF, self).__init__(**kwargs)
        self.units = len(adjlist)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Added for pcn
        self.share_weights = share_weights
        self.adjlist = adjlist
        assert len(adjlist) >= 1

    def gen_adjmat_and_reshapemat(self, input_dim):
        # Generate adjmat
        count = 0
        ilist = []
        jlist = []
        for adj in self.adjlist:
            for j in adj:
                ilist += [count]
                jlist += [j]
                count +=1
        self.num_edges_adjlist = count
        adjmat = sp.coo_matrix((np.ones(len(ilist), dtype=np.float32), (ilist, jlist)), shape=(self.num_edges_adjlist, input_dim), dtype=np.float32).tocsr()
        self.adjmat = K.variable(adjmat, dtype=None, name="adjmat", constraint=None)

        # Generate reshapemat
        count = 0
        ilist = []
        jlist = []
        for i, adj in enumerate(self.adjlist):
            for _ in adj:
                ilist += [i]
                jlist += [count]
                count+=1
        reshapemat = sp.coo_matrix((np.ones(len(ilist), dtype=np.float32), (ilist, jlist)), shape=(self.units, self.num_edges_adjlist), dtype=np.float32).tocsr()
        self.reshapemat = K.variable(reshapemat, dtype=None, name="reshapemat", constraint=None)

        print("PCN: Size {0} to size {1}".format(self.adjmat.shape[1], self.reshapemat.shape[0]))


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units) #(batch_dim, output_dim)


    def build(self, input_shape):
        if len(input_shape) == 2:
            self.input_features = 1
        elif len(input_shape) == 3:
            self.input_features = input_shape[2]

        input_nodes = input_shape[1]
        self.gen_adjmat_and_reshapemat(input_nodes)

        if self.share_weights:
            self.kernel = self.add_weight(shape=(self.num_edges_adjlist,),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.num_edges_adjlist,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None
        else:
            self.kernel = self.add_weight(shape=(self.input_features, self.num_edges_adjlist,),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.input_features, self.num_edges_adjlist,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None

        self.built = True

    def call(self, x):

        if len(x.shape)==2:
            x_a = K.expand_dims(x, axis=2)
        else:
            x_a = x
        features = []
        for i in range(self.input_features):
            xi = x_a[:,:,i]
            x_T = K.transpose(xi)
            flat_f_T = K.dot(self.adjmat, x_T)
            flat_f = K.transpose(flat_f_T)
            if self.share_weights:
                if self.use_bias:
                    flat_f2 = flat_f * self.kernel + self.bias
                else:
                    flat_f2 = flat_f * self.kernel
            else:
                if self.use_bias:
                    flat_f2 = flat_f * self.kernel[i] + self.bias[i]
                else:
                    flat_f2 = flat_f * self.kernel[i]

            flat_f2_T = K.transpose(flat_f2)
            f_out_T = K.dot(self.reshapemat, flat_f2_T)
            f_out = K.transpose(f_out_T)
            features += [f_out]
        output = K.concatenate(features, axis=2)

        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(PartialConnection_MF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class GraphConvolution(Layer):

    def __init__(self, output_features, adjMats,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(GraphConvolution, self).__init__(**kwargs)
        self.nodes = adjMats[0].shape[0]
        self.output_features = output_features
        self.adjMats = []
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = True

        self.support = len(adjMats)
        assert self.support >= 1

        for mat in adjMats:
            self.adjMats += [K.variable(mat, dtype=np.float32, name="adjmat", constraint=None)]



    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nodes, self.output_features) #(batch_dim, num_units, output_features)

    def build(self, input_shape):
        # (batch_dim, num_graphnodes, num_features)
        if len(input_shape) == 2:
            self.input_features = 1
        elif len(input_shape) == 3:
            self.input_features = input_shape[2]

        # self.adjCountMats = []
        # for adjmat in self.adjMats:
        #     countmat = 1/np.array(adjmat.sum(1)).reshape(-1,1)
        #     self.adjCountMats += [K.variable(countmat, dtype=np.float32, name="countmat", constraint=None)]

        
        self.kernel = self.add_weight(shape=(self.support, self.output_features, self.input_features),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_features,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True


    def call(self, x):
        if len(x.shape)==2:
            x_expand = K.expand_dims(x, axis=2)
            x_T = K.transpose(x_expand)
        else:
            x_T = K.transpose(x)



        # supports = []
        # for i in range(self.support):
        #     f_T = K.dot(self.adjMats[i], x_T)
        #     f_T = K.dot(f_T, self.kernel[i])
        #     res = K.transpose(f_T)
        #     supports += [res]
        supports = []
        for i in range(self.support):
            features = []
            for j in range(self.input_features):
                f_T = K.dot(self.adjMats[i], x_T[j])
                f = K.transpose(f_T)
                features += [f]

            if len(features)==1:
                feature = K.expand_dims(features[0], axis=2)
            else:
                feature = K.concatenate(features, axis=2)
            s = tf.tensordot(feature, self.kernel[i], axes=[2,1])
            supports += [s]
        # A = np.arange(18).reshape(3,2,3)
        # b = np.array([[1,2],[1,2],[1,2]])
        # A = K.variable(A)
        # b = K.variable(b)
        # R = tf.tensordot(A,b, axes=[2,1])
        # print(K.get_value(R))

        output = supports[0]
        for i in range(1, self.support):
            output += supports[i]


        if self.use_bias:
            output += self.bias
        return self.activation(output)


    def get_config(self):
        config = {'units': self.nodes,
                  'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PartialConnection(Layer):

    def __init__(self, adjlist, 
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(PartialConnection, self).__init__(**kwargs)
        self.units = len(adjlist)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # Added for pcn
        self.adjlist = adjlist
        assert len(adjlist) >= 1

    def gen_adjmat_and_reshapemat(self, input_dim):
        # Generate adjmat
        count = 0
        ilist = []
        jlist = []
        for adj in self.adjlist:
            for j in adj:
                ilist += [count]
                jlist += [j]
                count +=1
        self.num_edges_adjlist = count
        adjmat = sp.coo_matrix((np.ones(len(ilist), dtype=np.float32), (ilist, jlist)), shape=(self.num_edges_adjlist, input_dim), dtype=np.float32).tocsr()
        self.adjmat = K.variable(adjmat, dtype=None, name="adjmat", constraint=None)

        # Generate reshapemat
        count = 0
        ilist = []
        jlist = []
        for i, adj in enumerate(self.adjlist):
            for _ in adj:
                ilist += [i]
                jlist += [count]
                count+=1
        reshapemat = sp.coo_matrix((np.ones(len(ilist), dtype=np.float32), (ilist, jlist)), shape=(self.units, self.num_edges_adjlist), dtype=np.float32).tocsr()
        self.reshapemat = K.variable(reshapemat, dtype=None, name="reshapemat", constraint=None)

        print("PCN: Size {0} to size {1}".format(self.adjmat.shape[1], self.reshapemat.shape[0]))


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units) #(batch_dim, output_dim)


    def build(self, input_shape):
        input_dim = input_shape[1]
        self.gen_adjmat_and_reshapemat(input_dim)

        self.kernel = self.add_weight(shape=(self.num_edges_adjlist,),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_edges_adjlist,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, x):
        x_T = K.transpose(x)
        flat_f_T = K.dot(self.adjmat, x_T)
        flat_f = K.transpose(flat_f_T)
        if self.use_bias:
            flat_f2 = flat_f * self.kernel + self.bias
        else:
            flat_f2 = flat_f * self.kernel

        flat_f2_T = K.transpose(flat_f2)
        output_T = K.dot(self.reshapemat, flat_f2_T)
        output = K.transpose(output_T)

        return self.activation(output)

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(PartialConnection, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))