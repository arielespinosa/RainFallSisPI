'''
  Implementation of custom keras regression models.
  This is prepare for train in La Habana.
  That is why shape input is (375, 3)
'''
import os
import pickle
import numpy as np
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import (Dense, LSTM, Conv1D, BatchNormalization, Dropout, Flatten, Input,
						concatenate, Reshape, Add, GlobalAveragePooling1D, MaxPooling1D, Conv2D, UpSampling2D,
						MaxPooling2D, )
from keras.layers.noise import AlphaDropout

from keras.optimizers import Adam, SGD
from keras import regularizers
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.preprocessing import MinMaxScaler
from preprocess.file import write_serialize_file

from keras.layers.convolutional_recurrent import ConvLSTM2D
import settings as config


# Abstract RNA class
class RNA:

	def __init__(self, parameters = None):
		if isinstance(parameters, dict):
			self.dense_units=parameters["dense_units"]
			self.h_activation=parameters["h_activation"]
			self.o_activation=parameters["o_activation"]
			self.batch_norm=parameters["batch_norm"]
			self.dropout=parameters["dropout"]
			self.dropout_rate=parameters["dropout_rate"]
			#self.kernel_initializer=parameters["kernel_initializer"]
			self.optimizer=parameters["optimizer"]
			self.loss=parameters["loss"]
			self.metrics=parameters["metrics"]
			self.callbacks=parameters["callbacks"]
			self.shape=parameters["shape"]
			self.name = parameters["name"]
			self.model = None
			self.history = None
			#self.score = None

		elif isinstance(parameters, str):
			self.model = self.load(parameters)
			name = parameters.split("/")[-1].split(".")[0]
			self.name = name
		else:
			return None

	def train(self, x_train, y_train, validation_data=None, validation_split = 0.04, batch_size = 32, epochs = 32, workers = 6, use_multiprocessing = True, shuffle=True):	
		self.history = self.model.fit(x_train, y_train, validation_data=validation_data, validation_split=validation_split, batch_size=batch_size, epochs=epochs, workers=workers, 
						use_multiprocessing=use_multiprocessing, callbacks=self.callbacks, shuffle=shuffle)

	def train_generator(self, training_generator, validation_generator, epochs, workers = 4, use_multiprocessing = True):
		self.history = self.model.fit_generator(generator = training_generator, validation_data = validation_generator, callbacks=self.callbacks, workers=workers, use_multiprocessing=use_multiprocessing, epochs=epochs)
		
	def evaluate(self, x_test, y_test, batch_size):
		return self.model.evaluate(x_test, y_test, batch_size=batch_size)	

	def predict_generator(self, predict_generator, path_to_save = None, save_predict_list = False):
		results = self.model.predict_generator(generator = predict_generator, verbose = 1)
		
		# For save prediction results ready for plot
		results = predict_generator.y_scaler.inverse_transform(results)
		
		if save_predict_list:
			predict_generator.save_predict_files_list()

		if path_to_save is None:
			path=os.path.join(config.BASE_DIR, "rna/outputs/{}/predictions_{}.dat".format(self.name, self.name))
		
		write_serialize_file(results, path)

		#predict_generator.save_scalers()
		return results

	def predict(self, x_predict):
		results = self.model.predict(x_predict)
		return results
			
	def save(self, path = None):
		# Creates a HDF5 file in the path provided. Path must include file name and extension ('.h5')
		if path is None:
			path = os.path.join(config.BASE_DIR, "rna/outputs/{}/{}.h5".format(self.name, self.name))
		
		self.model.save(path)   

	def save_history(self, path = None):
		# Creates a HDF5 file in the path provided. Path must include file name and extension ('.h5')
		if path is None:
			path=os.path.join(config.BASE_DIR, "rna/outputs/{}/history_{}.dat".format(self.name, self.name))
			
		write_serialize_file(self.history, path)

	def load(self, path):
		# Load a model from HDF5 file. Path must include file_name and extension ('.h5')
		return load_model(path)

# Neuronal Support Vector Machines .
class NSVM(RNA):

	def __init__(self, parameters):
		super().__init__(parameters)

		if isinstance(parameters, dict):
			self.model = self.__create()
	
	# Working
	def __create(self):
		inputs, h_layers = [], []

		for i in range(3):
			inputs.append(Input(shape=self.shape))
			h_layers.append(Dense(1, activation=self.h_activation)(inputs[i]))

		# Joining layers h_layer1, h_layer2, h_layer3
		_layer = Add()(h_layers)
	
		# Hiddens layers
		hidden_layers = len(self.dense_units)

		if hidden_layers > 1:
			for i in range(hidden_layers):
				_layer = Dense(self.dense_units[i], activation=self.h_activation)(_layer)
					
				if self.dropout:		
					if self.dropout_rate[i] and self.dropout == "d":
						_layer = Dropout(self.dropout_rate[i])(_layer)
					elif self.dropout_rate[i] and self.dropout == "ad":
						_layer = AlphaDropout(self.dropout_rate[i])(_layer)

				if self.batch_norm and self.batch_norm[i]:
					_layer = BatchNormalization(momentum=self.batch_norm[i][0], epsilon=self.batch_norm[i][1])(_layer)
		else:
			_layer = Dense(self.dense_units[0], activation=self.h_activation)(_layer)
		
		# Output layer
		_output = Dense(1, activation = self.o_activation)(_layer)

		model = Model(inputs=inputs, outputs=_output, name = self.name)
		model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

		return model

# Multi Layer Perceptron neuronal network		
class MLP(RNA):
	
	def __init__(self, parameters):
		super().__init__(parameters)
		
		if isinstance(parameters, dict):
			self.model = self.__create()
			#self.model = self.__create_generalized()
			#self.model = self.__create_sequential()

	# Working
	def __create(self):

		_input = Input(shape=self.shape)

		try:		
			_layer = Flatten()(_input)
		except:
			_layer = _input

		hidden_layers = len(self.dense_units)

		if hidden_layers > 1:
			for i in range(hidden_layers):
				_layer = Dense(self.dense_units[i], activation=self.h_activation)(_layer)

				if self.dropout:		
					if self.dropout_rate[i] and self.dropout == "d":
						_layer = Dropout(self.dropout_rate[i])(_layer)
					elif self.dropout_rate[i] and self.dropout == "ad":
						_layer = AlphaDropout(self.dropout_rate[i])(_layer)

				if self.batch_norm and self.batch_norm[i]:
					_layer = BatchNormalization(momentum=self.batch_norm[i][0], epsilon=self.batch_norm[i][1])(_layer)
		else:
			_layer = Dense(self.dense_units[0], activation=self.h_activation)(_layer)
	
		output = Dense(1, activation = self.o_activation)(_layer)
		
		model = Model(inputs = _input, outputs = output, name = self.name)
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		return model
		
	def __create_generalized(self):

		i_layer = Input(shape=self.shape, name="i_layer")
		
		h_layer_1 = Dense(5, activation="sigmoid", name="h_layer_1")(i_layer)
		h_layer_2 = Dense(5, activation="sigmoid", name="h_layer_2")(concatenate([i_layer, h_layer_1]))
		
		o_layer = Dense(1, activation="sigmoid", name="0_layer")(concatenate([i_layer, h_layer_1, h_layer_2]))
			
		model = Model(inputs = i_layer, outputs = o_layer)
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		return model

		"""
		_input = Input(shape=self.shape)
		_layer = Flatten()(_input)

		hidden_layers = len(self.dense_units)

		if hidden_layers > 1:
			for i in range(hidden_layers):
				_layer = Dense(self.dense_units[i], activation=self.h_activation)(_layer)

				if self.dropout:		
					if self.dropout_rate[i] and self.dropout == "d":
						_layer = Dropout(self.dropout_rate[i])(_layer)
					elif self.dropout_rate[i] and self.dropout == "ad":
						_layer = AlphaDropout(self.dropout_rate[i])(_layer)

				if self.batch_norm and self.batch_norm[i]:
					_layer = BatchNormalization(momentum=self.batch_norm[i][0], epsilon=self.batch_norm[i][1])(_layer)
			
		output = Dense(self.shape[0], activation = self.o_activation)(_layer)
		
		model = Model(inputs = _input, outputs = output, name = self.name)
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		return model
		"""
		

# Long-Short Tensor Memory neuronal network	
class LSTMNN(RNA):

	def __init__(self, parameters):
		super().__init__(parameters)
		
		if isinstance(parameters, dict):
			self.model = self.__create_sequential()
			#self.model = self.__create()

	def __create(self):
		_input = Input(shape=self.shape)

		_layer = _input

		hidden_layers = len(self.dense_units)

		if hidden_layers > 1:
			for i in range(hidden_layers):
				_layer = LSTM(self.dense_units[i], activation=self.h_activation)(_layer)

				if self.dropout:		
					if self.dropout_rate[i] and self.dropout == "d":
						_layer = Dropout(self.dropout_rate[i])(_layer)
					elif self.dropout_rate[i] and self.dropout == "ad":
						_layer = AlphaDropout(self.dropout_rate[i])(_layer)

				if self.batch_norm and self.batch_norm[i]:
					_layer = BatchNormalization(momentum=self.batch_norm[i][0], epsilon=self.batch_norm[i][1])(_layer)
		else:
			_layer = Dense(self.dense_units[0], activation=self.h_activation)(_layer)
	
		output = Dense(1, activation = self.o_activation)(_layer)
		
		model = Model(inputs = _input, outputs = output, name = self.name)
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		return model
		
	def __create_multiple_inputs(self):

		inputs = []
		outputs = []

		for i in range(375):
			i_layer = Input(shape=(3, ))
			h_layer = LSTM(1)(i_layer)
			h_layer = Dense(8, activation="relu")(i_layer)
			#h_layer = LSTM(32)(x)
			o_layer = Dense(1, activation="softmax")(h_layer)
			

			inputs.append(i_layer)
			outputs.append(o_layer)

		main_output_layer = keras.layers.add(outputs)
		#main_output_layer = Dense(375, activation = "sigmoid")(main_output_layer)
		
		model = Model(inputs = inputs, outputs = main_output_layer, name = "pepe")
		model.compile(Adam(), loss="mse", metrics=["accuracy", "mse", "mae"])
		return model

		"""
		_input = Input(shape=self.shape)
		_layer = Flatten()(_input)

		hidden_layers = len(self.dense_units)

		if hidden_layers > 1:
			for i in range(hidden_layers):
				_layer = Dense(self.dense_units[i], activation=self.h_activation)(_layer)

				if self.dropout:		
					if self.dropout_rate[i] and self.dropout == "d":
						_layer = Dropout(self.dropout_rate[i])(_layer)
					elif self.dropout_rate[i] and self.dropout == "ad":
						_layer = AlphaDropout(self.dropout_rate[i])(_layer)

				if self.batch_norm and self.batch_norm[i]:
					_layer = BatchNormalization(momentum=self.batch_norm[i][0], epsilon=self.batch_norm[i][1])(_layer)
			
		output = Dense(self.shape[0], activation = self.o_activation)(_layer)
		
		model = Model(inputs = _input, outputs = output, name = self.name)
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		return model
		"""
	
	# Working
	def __create_sequential(self):

		model = Sequential()
		model.add(LSTM(units=160, batch_input_shape=self.shape, return_sequences=True))
		model.add(LSTM(units=80))		
		model.add(BatchNormalization())
		model.add(Dense(1, activation=self.o_activation))
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		return model

	# Working
	def __create_sequential3(self):

		model = Sequential()
		model.add(LSTM(units=self.dense_units[0], batch_input_shape=(None, 3, 1), return_sequences=False))

		hidden_layers = len(self.dense_units)

		if hidden_layers > 1:
			for i in range(hidden_layers):
				model.add(LSTM(units=self.dense_units, return_sequences=False))
		else:
			model.add(LSTM(units=self.dense_units, return_sequences=False))
		model.add(Dense(1))
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		return model

	def __create_sequentia_2(self):

		model = Sequential()
		#model.add(LSTM(units=900, batch_input_shape=(None, 375, 3), return_sequences=False))
		model.add(Conv1D(256, (1, ), input_shape=(5, 375), padding='same', use_bias=False))
		model.add(LSTM(units=900, batch_input_shape=(None, 5, 375), return_sequences=False, use_bias=False))
		#model.add(LSTM(units=900, return_sequences=False))
		model.add(Dense(375, use_bias=False))
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		return model

# Convolutional neuronal network	
class CONV(RNA):

	def __init__(self, parameters):
		super().__init__(parameters)

		if isinstance(parameters, dict):
			self.model = self.__create_sequential()
			#self.model = self.__convlstm_2()
			#self.model = self.__conv_encode_lstm()
			#self.model = self.__conv_autoencoder()

	def __create(self):
		_input = Input(shape=self.shape)
		#_layer = Flatten()(_input)
		_layer = LSTM(800, input_shape=(1, 1))(_input)

		#_layer = Flatten()(_input)

		"""
		hidden_layers = len(self.dense_units)

		if hidden_layers > 1:
			for i in range(hidden_layers):
				_layer = Dense(self.dense_units[i], activation=self.h_activation)(_layer)

				if self.dropout:		
					if self.dropout_rate[i] and self.dropout == "d":
						_layer = Dropout(self.dropout_rate[i])(_layer)
					elif self.dropout_rate[i] and self.dropout == "ad":
						_layer = AlphaDropout(self.dropout_rate[i])(_layer)

				if self.batch_norm and self.batch_norm[i]:
					_layer = BatchNormalization(momentum=self.batch_norm[i][0], epsilon=self.batch_norm[i][1])(_layer)
		
		"""
		output = Dense(self.shape[0], activation = self.o_activation)(_layer)
		
		model = Model(inputs = _input, outputs = output, name = self.name)
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		return model
		
	# Working
	def __create_sequential(self):
		model = Sequential()
		model.add(Conv1D(32, kernel_size=3, activation=self.h_activation, input_shape=self.shape,	padding='same'))
		model.add(Conv1D(32, kernel_size=3, activation=self.h_activation, padding='same'))
		model.add(MaxPooling1D(3))
		model.add(Conv1D(64, kernel_size=3, activation=self.h_activation, padding='same'))
		model.add(Conv1D(64, kernel_size=3, activation=self.h_activation, padding='same'))
		model.add(GlobalAveragePooling1D())
		model.add(Dropout(0.5))
		model.add(Dense(1, activation=self.o_activation))
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
				
		"""
		model.add(ConvLSTM2D(filters=375, kernel_size=(3, 3), input_shape=(None, 15, 25, 1), padding='same', return_sequences=True))
		model.add(BatchNormalization())

		model.add(ConvLSTM2D(filters=375, kernel_size=(3, 3), padding='same', return_sequences=True))
		model.add(BatchNormalization())

		model.add(ConvLSTM2D(filters=375, kernel_size=(3, 3),padding='same', return_sequences=True))
		model.add(BatchNormalization())

		model.add(ConvLSTM2D(filters=375, kernel_size=(3, 3), padding='same', return_sequences=True))
		model.add(BatchNormalization())

		model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)
		"""

		return model

	# Working
	def __convlstm_2(self):

		model = Sequential()

		model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), input_shape=(5, 15, 25, 1), padding='same', return_sequences=True,  activation='tanh', 
						recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, 
						dropout=0.3, recurrent_dropout=0.3, go_backwards=True ))
		model.add(BatchNormalization())

		model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True,  activation='tanh', 
						recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, 
						dropout=0.3, recurrent_dropout=0.3, go_backwards=True))
		model.add(BatchNormalization())

		model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3),padding='same', return_sequences=True,  activation='tanh', 
						recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, 
						dropout=0.3, recurrent_dropout=0.3, go_backwards=True))
		model.add(BatchNormalization())

		model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', activation='tanh', 
						recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform', unit_forget_bias=True, 
						dropout=0.3, recurrent_dropout=0.3, go_backwards=True))
		model.add(BatchNormalization())

		model.add(Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same'))

		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

		return model

	# Keras bug
	def __conv_encode_lstm(self):

		input_layer = Input(shape=(5, 15, 25, 1))

		enc_1 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_sequences=True, return_state=True, name="enc_convlstm_1")
		encod_1_outputs, encod_1_h_state, encod_1_c_state = enc_1(input_layer)

		enc_2 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_state=True)
		encod_2_outputs, encod_2_h_state, encod_2_c_state = enc_2(encod_1_outputs, initial_state=[encod_1_h_state, encod_1_c_state])

		# Forecasting
		forecasting_1 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_sequences=True, name="forecasting_1")
		forecasting_1 = forecasting_1(input_layer, initial_state=[encod_1_h_state, encod_1_c_state])
		
		forecasting_2 = ConvLSTM2D(8, (3, 3), activation='relu', padding='same', name="forecasting_2")
		forecasting_2 = forecasting_2(forecasting_1, initial_state=[encod_2_h_state, encod_2_c_state])

		join_outputs = keras.layers.concatenate([encod_2_outputs, forecasting_2])

		output = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name="outputs")
		final_output = output(join_outputs)

		# Encoder
		"""

		enc_1 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_sequences=True, return_state=True, name="enc_convlstm_1")
		encod_1_outputs, encod_1_h_state, encod_1_c_state = enc_1(input_layer)

		enc_2 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_state=True)
		encod_2_outputs, encod_2_h_state, encod_2_c_state = enc_2(encod_1_outputs, initial_state=[encod_1_h_state, encod_1_c_state])

		# Forecasting
		forecasting_1 = ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_sequences=True, name="forecasting_1")
		forecasting_1 = forecasting_1(input_layer, initial_state=[encod_1_h_state, encod_1_c_state])
		
		forecasting_2 = ConvLSTM2D(8, (3, 3), activation='relu', padding='same', name="forecasting_2")
		forecasting_2 = forecasting_2(forecasting_1, initial_state=[encod_2_h_state, encod_2_c_state])

		join_outputs = keras.layers.concatenate([encod_2_outputs, forecasting_2])

		output = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name="outputs")
		final_output = output(join_outputs)

		"""
	
		model = Model(input_layer, final_output)
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

		return model

		print(forecasting_1)
		time.sleep(345)

	# Experiment
	def __conv_autoencoder(self):

		input_layer = Input(shape=(28, 28, 1))

		x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
		x = MaxPooling2D((2, 2), padding='same')(x)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
		x = MaxPooling2D((2, 2), padding='same')(x)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
		encoded = MaxPooling2D((2, 2), padding='same')(x)

		x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
		x = UpSampling2D((2, 2))(x)
		x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
		x = UpSampling2D((2, 2))(x)
		x = Conv2D(16, (3, 3), activation='relu')(x)
		x = UpSampling2D((2, 2))(x)
		decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
	
		model = Model(input_layer, decoded)
		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

		return model

# Extreme Machine Learning neuronal network
class ELM(RNA):

	def __init__(self, parameters):
		super().__init__(parameters)

		if isinstance(parameters, dict):
			self.model = self.__create_sequential()
	
	# Working
	def __create_sequential(self):

		model = Sequential()
		model.add(Dense(self.dense_units, trainable=False, activation=self.h_activation, input_shape=self.shape))
		
		model.add(Dense(1, activation=self.o_activation))

		model.compile(self.optimizer, loss=self.loss, metrics=self.metrics)

		return model
