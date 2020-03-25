from keras.models import *
from keras.layers import *
import keras.backend as K
from src.layers import partial, graph

# def create_tetrahedra_network(adjLists_forPCN, adjMats_forGCN, shape, name = 'tetranet'):
def create_tetrahedra_network(adjLists_forPCN, shape, name = 'tetranet'):


	input = Input(shape=shape)

	# 256*256 to 128*128
	# cnv1_ = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu') (input)
	# cnv1 = BatchNormalization()(cnv1_)
	# cnv2_ = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu') (cnv1)
	# cnv2 = BatchNormalization()(cnv2_)
	cnv2_ = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu') (input)
	cnv2 = BatchNormalization()(cnv2_)

	# 128*128 to 64*64
	r1 = bottleneck2D(cnv2, 128)
	pool = MaxPool2D(pool_size=(2,2), strides=(2,2))(r1)

	r4 = bottleneck2D(pool, 128)
	r5 = bottleneck2D(r4, 128)
	r6 = bottleneck2D(r5, 256)

	# hg1
	hg1 = hourglass(r6, 4, 512)
	l1 = Conv2D(512, kernel_size=1, padding='same', activation='relu')(hg1)
	l1 = BatchNormalization()(l1)
	l2 = Conv2D(256, kernel_size=1, padding='same', activation='relu')(l1)
	l2 = BatchNormalization()(l2)

	out1 = Conv2D(8, kernel_size=1, name='hg_out1', activation = 'sigmoid')(l2)
	out1_ = Conv2D(256 + 128, padding='same', kernel_size=1)(out1)

	cat1 = Concatenate()([l2, pool])
	cat1_ = Conv2D(256 + 128, padding='same', kernel_size=1)(cat1)
	int1 = Add()([cat1_, out1_])

	# # hg2
	# hg2 = hourglass(int1, 4, 512)
	# l3 = Conv2D(512, kernel_size=1, padding='same', activation='relu')(hg2)
	# l3 = BatchNormalization()(l3)
	# l4 = Conv2D(256, kernel_size=1, padding='same', activation='relu')(l3)
	# l4 = BatchNormalization()(l4)

	# out2 = Conv2D(scale * 64, kernel_size=1, name='hg_out2', activation = 'sigmoid')(l4)
	# out2_ = Conv2D(256 + 256, padding='same', kernel_size=1)(out2)

	# cat2 = Concatenate()([l4, l2])
	# cat2_ = Conv2D(256 + 256, padding='same', kernel_size=1)(cat2)
	# int2 = Add()([cat2_, out2_])


	# last part
	# tsdf = lastpart(int1, adjLists_forPCN, adjMats_forGCN)
	tsdf = lastpart(int1, adjLists_forPCN)

	return Model(inputs=input, outputs=tsdf, name = name)


# def lastpart(bottom, adjLists_forPCN, adjMats_forGCN):
def lastpart(bottom, adjLists_forPCN):

	# x64 -> x32
	c1 = Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(bottom)
	b1 = BatchNormalization()(c1)
	p1 = MaxPool2D(pool_size=(2,2), strides=(2,2))(b1)

	# x16
	c2 = Conv2D(1024, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(p1)
	b2 = BatchNormalization()(c2)
	p2 = MaxPool2D(pool_size=(2,2), strides=(2,2))(b2)
	
	# x8
	c3 = Conv2D(2048, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(p2)
	b3 = BatchNormalization()(c3)
	p3 = MaxPool2D(pool_size=(2,2), strides=(2,2))(b3)

	# x4
	c4 = Conv2D(2048, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(p3)
	b4 = BatchNormalization()(c4)
	p4 = MaxPool2D(pool_size=(2,2), strides=(2,2))(b4)
	
	# x2
	c5 = Conv2D(2048, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(p4)
	b5 = BatchNormalization()(c5)
	p5 = MaxPool2D(pool_size=(2,2), strides=(2,2))(b5)
	f1 = Flatten()(p5)

	f2 = Dense(max(max(adjLists_forPCN[0])) + 1, activation="relu")(f1)
	
	# tsdf = iterateGCN_PCN(f2, adjLists_forPCN, adjMats_forGCN[1:], 0, "relu")
	tsdf = iteratePCN(f2, adjLists_forPCN, 0, "relu")


	# gc = partial.GraphConvolution(1, adjMats_forGCN[0], activation="relu")(f2)

	# gc_out = iterateGCN_PCN(gc, adjLists_forPCN, adjMats_forGCN[1:], 0,"relu")

	# tsdf = Flatten()(gc_out)

	return tsdf

# def iterateGCN_PCN(bottom, adjLists_forPCN, adjMats_forGCN, count, activation):
def iteratePCN(bottom, adjLists_forPCN, count, activation):

	adjlist = adjLists_forPCN[count]
	# adjmats = adjMats_forGCN[count]
	
	# gc = graph.GraphConvolution(len(adjmat_forgcn), support=2, activation='relu')([pc]+adjmat_forgcn)
	# pc = Dense(adjmat_forgcn[0].shape[0], activation="relu")(bottom)
	# pc = partial.PartialConnection_MF(adjlist, share_weights=True, activation=activation)(bottom)
	pc = partial.PartialConnection(adjlist, activation=activation)(bottom)
	# gc = partial.GraphConvolution(1, adjmats, activation=activation)(pc)
	# gc = partial.PartialConnection(bottom, adjlist, activation="relu")
	
	if count < len(adjLists_forPCN)-2:
		# gc = iterateGCN_PCN(gc, adjLists_forPCN, adjMats_forGCN, count+1, "relu")
		pc = iteratePCN(pc, adjLists_forPCN, count+1, "relu")
	elif count == len(adjLists_forPCN)-2:
		# gc = iterateGCN_PCN(gc, adjLists_forPCN, adjMats_forGCN, count+1, "sigmoid")
		pc = iteratePCN(pc, adjLists_forPCN, count+1, "sigmoid")
	# if count == 0:
		# gcfinal = graph.GraphConvolution(len(adjmat), support=2, activation='relu')([pc]+adjmat_forgcn)
		# return gcfinal
	
	return pc



def hourglass(bottom, num_skip, num_channels):

	# Skip connection
	sk1 = bottleneck2D(bottom, num_channels)

	# Residual section
	pool = MaxPool2D(pool_size=2)(bottom)
	low1 = bottleneck2D(pool, num_channels)
	if num_skip > 1:
		low2 = hourglass(low1, num_skip - 1, num_channels)
	else:
		low2 = bottleneck2D(low1, num_channels)
	
	low3 = bottleneck2D(low2, num_channels)
	up2 = UpSampling2D(size = 2)(low3)

	return Add()([up2, sk1])


def hourglass_paper_implement(bottom, num_skip, num_channels):
	#The number of features is consistent across the whole hourglass.
	
	# Convolutional and max pooling layers are used to process features down to a very low resolution.
	up1 = bottleneck2D(bottom, num_channels)

	# At each max pooling step, the network branches oﬀ and applies more convolutions at the original pre-pooled resolution. 
	sk1 = bottleneck2D(up1, num_channels) # <- Branch off and apply more convolutions before pooling.
	pool = MaxPool2D(pool_size=2)(up1)

	# Residual section
	if num_skip > 1:
		low3 = hourglass_paper_implement(pool, num_skip - 1, num_channels)
	else:
		low1 = bottleneck2D(pool, num_channels)
		low2 = bottleneck2D(low1, num_channels)
		low3 = bottleneck2D(low2, num_channels)
	
	up2 = UpSampling2D(size = 2)(low3)
	up3 = Add()([up2, sk1])
	out = bottleneck2D(up3, num_channels)

	return out


def hourglass_large(bottom, num_skip, num_channels):

	# Skip connection
	up1 = bottleneck2D(bottom, num_channels)
	up2 = bottleneck2D(up1, num_channels)
	up4 = bottleneck2D(up2, num_channels)

	# Residual section
	pool = MaxPool2D(pool_size=2)(bottom)
	low1 = bottleneck2D(pool, num_channels)
	low2 = bottleneck2D(low1, num_channels)
	low3 = bottleneck2D(low2, num_channels)
	if num_skip > 1:
		low6 = hourglass_large(low3, num_skip - 1, num_channels)
	else:
		low4 = bottleneck2D(low3, num_channels)
		low5 = bottleneck2D(low4, num_channels)
		low6 = bottleneck2D(low5, num_channels)
	low7 = bottleneck2D(low6, num_channels)
	up5 = UpSampling2D(size = 2)(low7)

	return Add()([up4, up5])


def hourglass_large_paper_implement(bottom, num_skip, num_channels):
	#The number of features is consistent across the whole hourglass.
	
	# Convolutional and max pooling layers are used to process features down to a very low resolution.
	up1 = bottleneck2D(bottom, num_channels)
	up2 = bottleneck2D(up1, num_channels)
	up3 = bottleneck2D(up2, num_channels)
	pool = MaxPool2D(pool_size=2)(up3)

	# At each max pooling step, the network branches oﬀ and applies more convolutions at the original pre-pooled resolution. 
	sk1 = bottleneck2D(up3, num_channels) # <- Branch off and apply more convolutions before pooling.
	sk2 = bottleneck2D(sk1, num_channels)
	sk3 = bottleneck2D(sk2, num_channels)

	# Residual section
	if num_skip > 1:
		low3 = hourglass_large_paper_implement(pool, num_skip - 1, num_channels)
	else:
		low1 = bottleneck2D(pool, num_channels)
		low2 = bottleneck2D(low1, num_channels)
		low3 = bottleneck2D(low2, num_channels)
	
	up4 = UpSampling2D(size = 2)(low3)
	up3 = Add()([up4, sk3])
	out = bottleneck2D(up3, num_channels)

	return out


def bottleneck2D(bottom, num_out_channels):
	# skip layer
	if K.int_shape(bottom)[-1] == num_out_channels:
		_skip = bottom
	else:
		# linear activation
		_skip = Conv2D(num_out_channels, kernel_size=1, activation='linear', padding='same') (bottom)

	# residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
	_x = Conv2D(int(num_out_channels/2), kernel_size=1, activation='relu', padding='same') (bottom)
	_x = BatchNormalization()(_x)
	_x = Conv2D(int(num_out_channels/2), kernel_size=3, activation='relu', padding='same') (_x)
	_x = BatchNormalization()(_x)
	_x = Conv2D(num_out_channels, kernel_size=1, activation='relu', padding='same') (_x)
	_x = BatchNormalization()(_x)
	_x = Add()([_skip, _x])

	return _x