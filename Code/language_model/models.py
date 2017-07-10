import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Merge
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Input, Embedding, LSTM, Dense, merge, SimpleRNN, TimeDistributedDense


class RNNModel:
	def getModel(self, params, weight=None  ):
		
		lstm_cell_size = params['lstm_cell_size']
		print "params['embeddings_dim'] = ", params['embeddings_dim']
		print "lstm_cell_size= ", lstm_cell_size
		inp = Input(shape=(params['inp_length'],), dtype='int32', name="inp")
		embedding = Embedding(input_dim = params['vocab_size']+1, output_dim = params['embeddings_dim'],
			input_length = params['inp_length'],
			dropout=0.2,
			mask_zero=True,
			trainable=True) (inp)
		lstm_out = LSTM(lstm_cell_size, return_sequences=True)(embedding)
		out = TimeDistributedDense(params['vocab_size'], activation='softmax')(lstm_out)
		model = Model(input=[inp], output=[out])
		model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
		print model.summary()

		return model

	
