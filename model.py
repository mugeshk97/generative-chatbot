import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


tokenizer = pickle.load(open('Asset/Tokenizer/tokenizer.pickle', 'rb'))
vocab_size = len(tokenizer.word_index) + 1
embedding_vector = pickle.load(open('Asset/Embedding/embedding_weights.pkl', 'rb'))

# Encoder
inference_encoder_inputs = Input(shape=(None,))
embedded_encoder_inputs = Embedding(input_dim=vocab_size, output_dim=200, input_length=20, weights=[embedding_vector])(
    inference_encoder_inputs)
encoder_output, state_h, state_c = LSTM(1024, return_state=True, return_sequences=True)(embedded_encoder_inputs)
encoder_states = [state_h, state_c]
# Decoder
inference_decoder_inputs = Input(shape=(None,))
embedded_decoder_inputs = Embedding(input_dim=vocab_size, output_dim=200, input_length=20, weights=[embedding_vector])(
    inference_decoder_inputs)
decoder_lst_layer = LSTM(1024, return_state=True, return_sequences=True)
decoder_output, _, _ = decoder_lst_layer(embedded_decoder_inputs, initial_state=encoder_states)
decoder_output = K.reshape(decoder_output, (-1, decoder_output.shape[2]))
dense_layer = Dense(vocab_size, activation='softmax')
decoder_output = dense_layer(decoder_output)

model = Model([inference_encoder_inputs, inference_decoder_inputs], decoder_output)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

model.load_weights('Asset/Model/final_weights.h5')


def inference_model():
    inference_encoder_model = Model(inference_encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(1024,))
    decoder_state_input_c = Input(shape=(1024,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_output, state_h, state_c = decoder_lst_layer(embedded_decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_output = dense_layer(decoder_output)
    inference_decoder_model = Model([inference_decoder_inputs] + decoder_states_inputs,
                                    [decoder_output] + decoder_states)
    return inference_encoder_model, inference_decoder_model
