from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model,model_from_json
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#from keras.callbacks import TensorBoard
from keras.callbacks import TensorBoard
import tensorflow as tf
from keras.regularizers import l2

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g


# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g


# define the standalone generator model
def define_autoencoder(image_shape=(256,256,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	e1 = define_encoder_block(in_image, 64, batchnorm=False)
	e2 = define_encoder_block(e1, 128)
	e3 = define_encoder_block(e2, 256)
	e4 = define_encoder_block(e3, 512)
	e5 = define_encoder_block(e4, 512)
	e6 = define_encoder_block(e5, 512)
	e7 = define_encoder_block(e6, 512)
	# bottleneck, no batch norm and relu
	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
	b = Activation('relu')(b)
	# decoder model
	d1 = decoder_block(b, e7, 512)
	d2 = decoder_block(d1, e6, 512)
	d3 = decoder_block(d2, e5, 512)
	d4 = decoder_block(d3, e4, 512, dropout=False)
	d5 = decoder_block(d4, e3, 256, dropout=False)
	d6 = decoder_block(d5, e2, 128, dropout=False)
	d7 = decoder_block(d6, e1, 64, dropout=False)
	# output
	g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model




def define_generators(bsize,seed,train_in_path,train_out_path,val_in_path,val_out_path):



	#we create two instances with the same arguments
	data_gen_args = dict(rotation_range=90,
						  width_shift_range=0.2,
						  height_shift_range=0.2,
						  rescale=1./255,
						  shear_range=0.2,
						  zoom_range=0.2)



	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)



	# Provide the same seed and keyword arguments to the fit and flow methods
	#-----train
	image_generator = image_datagen.flow_from_directory(
		train_in_path,
		class_mode=None,
		target_size=(256, 256),
		color_mode="grayscale",
		batch_size=bsize,
		shuffle=True,
		seed=seed)



	mask_generator = mask_datagen.flow_from_directory(
		train_out_path,
		class_mode=None,
		target_size=(256, 256),
		color_mode="grayscale",
		batch_size=bsize,
		shuffle=True,
		seed=seed)


	# combine generators into one which yields image and masks
	train_generator = zip(image_generator, mask_generator)


	# Provide the same seed and keyword arguments to the fit and flow methods
	#-----validation
	image_generator2 = image_datagen.flow_from_directory(
		val_in_path,
		class_mode=None,
		target_size=(256, 256),
		color_mode="grayscale",
		batch_size=bsize,
		shuffle=True,
		seed=seed)



	mask_generator2 = mask_datagen.flow_from_directory(
		val_out_path,
		class_mode=None,
		target_size=(256, 256),
		color_mode="grayscale",
		batch_size=bsize,
		shuffle=True,
		seed=seed)


	# combine generators into one which yields image and masks
	validation_generator = zip(image_generator2, mask_generator2)
	
return train_generator,validation_generator








if __name__ == '__main__':

	#-------------parameters

	seed = 1
	bsize=16

	train_in_path='/train_input_images/'
	train_out_path='/train_output_images/'
	val_in_path='/validation_input_images/'
	val_out_path='/validation_input_images/'

	#----------------------------------------generators defination

	train_generator,validation_generator=define_generators(bsize,seed,train_in_path,train_out_path,val_in_path,val_out_path)



	#----------------------------------------Train parameters and model definition

	#opt='adadelta'
	#opt='adam'
	epochn=10
	#lr=0.001

	#opt=tf.train.GradientDescentOptimizer()
	#opt = Adam(lr=0.01)
	opt='SGD'

	model=define_autoencoder((256,256,1))

	#----------------------------------------pretrained by ours

	# load json and create model
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	# load weights into new model
	model.load_weights("model.05-0.01.h5")



	#----------------------------------------Train the model

	model.compile(optimizer='adadelta', loss='mse')



	# simple early stopping
	# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=0.002)
	#cp = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', mode='min', verbose=1)
	cp = ModelCheckpoint(filepath='model.{epoch:02d}.h5', mode='min', verbose=1)



	model.fit_generator(
		train_generator,
		shuffle=True,
		steps_per_epoch=157000/bsize,
		epochs=epochn,
		callbacks=[cp])
		

	# validation_data=validation_generator,
	# validation_steps=15700,




	#-----------------------------------------Saving model
		
	# serialize model to JSON
	model_json = model.to_json()
	with open('model.json', 'w') as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights('model.h5')
	print('Saved model to disk')    

	# model saving
	model.save('final_model.h5')    


