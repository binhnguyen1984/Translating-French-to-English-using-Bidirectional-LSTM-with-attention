from __future__ import print_function

import tensorflow as tf
from keras.models import Model
import numpy as np
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply, Reshape
from keras.layers import RepeatVector, Dense, Activation, Lambda, Dropout
from keras.optimizers import Adam
from keras.engine.base_layer import Layer
from keras import backend as K

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra-eng/fra.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

def encode_text(input_text, max_length, num_tokens, token_index):
    input_data = np.zeros((max_length, num_tokens))
    for t,char in enumerate(input_text):
        input_data[t, token_index[char]] =1.
    return input_data

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    encoder_input_data[i] = encode_text(input_text,max_encoder_seq_length, num_encoder_tokens, input_token_index)
    decoder_input_data[i] = encode_text(target_text, max_decoder_seq_length, num_decoder_tokens, target_token_index)
    for t, char in enumerate(target_text):
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


class ExtractAxis(Layer):
    """
    Extract a certain axis with certain index while keeping other axes unchanged
    # Arguments:
       axis: dimention to extract
       index: index in the extracted axis
    """
    def __init__(self, axis, index, **kwargs):
        super(ExtractAxis, self).__init__(**kwargs)
        self.axis = axis
        self.index = index
        
    def call(self, inputs):
        return tf.gather(inputs, indices=self.index, axis=self.axis)
    
    def get_config(self):
        config= {'index': self.index, 'axis':self.axis}
        base_config = super(ExtractAxis, self).get_config()
        return dict(list(base_config.items())+ list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return tuple(input_shape[a] for a in range(len(input_shape)) if a!=self.axis)

class Attention(Layer):
    def __init__(self, encoder, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.encoder = encoder
        
    def call(self,inputs):
        a = self.encoder(inputs[0])
        context = one_step_attention(a, inputs[1])
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1]*2)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    repeator = RepeatVector(max_encoder_seq_length)
    concatenator = Concatenate(axis=-1)
    densor = Dense(max_encoder_seq_length, activation='tanh')
    activator = Activation('softmax')
    dotor = Dot(axes = 1)

    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([s_prev,a])
    e = densor(concat)
    alphas = activator(e)
    context = dotor([alphas,a])
    return context


# Define an input sequence and process it.

n_a = latent_dim
n_s = latent_dim

# encoder
encoder_output = Bidirectional(LSTM(n_a, return_sequences=True, dropout=0.5), input_shape=(max_encoder_seq_length, num_encoder_tokens))
attention = Attention(encoder_output)

# decoder 
decoder_lstm = LSTM(n_s, return_state = True, dropout=0.5)
output_layer = Dense(num_decoder_tokens, activation='softmax')

def train_model(Tx, Ty,num_encoder_tokens, num_decoder_tokens):
    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, num_encoder_tokens))
    Y = Input(shape=(Ty, num_decoder_tokens))
    s0 = Input(shape=(n_s,))
    c0 = Input(shape=(n_s,))
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []
    for t in range(Ty):
        context = attention([X,s])    
        extract_axis = ExtractAxis(axis=1, index=t)
        ground_true = extract_axis(Y)
        repeat = RepeatVector(context.shape[1].value)(ground_true)
        concat = Concatenate(axis=-1)([context, repeat])
       # initial_state = [hidden state, cell state]
        s, _, c = decoder_lstm(concat,initial_state = [s, c])
        
        # Apply Dense layer to the hidden state output of the decoder-attention LSTM
        out = output_layer(s)
        
        # Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
    
    # Create model instance taking three inputs and returning the list of outputs.
    model = Model(inputs = [X,s0,c0,Y], outputs = outputs)
        
    return model

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = train_model(max_encoder_seq_length, max_decoder_seq_length, num_encoder_tokens, num_decoder_tokens)
s0 = np.zeros((len(input_texts), n_s))
c0 = np.zeros((len(input_texts), n_s))

outputs = list(decoder_target_data.swapaxes(0,1))
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Run training
model.fit([encoder_input_data, s0, c0, decoder_input_data], outputs, epochs=epochs, batch_size=batch_size,
          validation_split=0.2)

# Save model
#model.save('att.h5')

reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decoder_model(n_s, max_length, num_tokens):
    Y = Input(shape=(max_length, num_tokens+n_s*2))
    s0 = Input(shape=(n_s,))
    c0 = Input(shape=(n_s,))
    s, _, c = decoder_lstm(Y,initial_state = [s0, c0])
    out = output_layer(s)
    model = Model(inputs = [Y,s0,c0], outputs = [out,s,c])
    return model



#model = load_model('s2s.h5')
# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

start = '\t'
y = np.zeros((1, num_decoder_tokens))
y[0,target_token_index[start]]=1.

X = Input(shape=(max_encoder_seq_length, num_encoder_tokens))
state = Input(shape=(n_s,), name='s')
context_model = Model([X,state], attention([X,state]))
Y = Input(shape=(max_decoder_seq_length, num_decoder_tokens+n_s*2))

decoder = decoder_model(n_s, max_encoder_seq_length, num_decoder_tokens)
def decode_sequence(input_seq):
    prediction = []
    Y0 = y
    s=s0
    c=c0
    for t in range(max_decoder_seq_length):
        context = context_model.predict([input_seq,s])    
        repeat = np.repeat(np.expand_dims(Y0, axis=1), context.shape[1], axis=1)
        concat = np.concatenate([context, repeat], axis=-1)
        out, s, c = decoder.predict([concat,s,c])
        sample_out_index = np.argmax(out, axis=-1)[0]
        sample_char = reverse_target_char_index[sample_out_index]
        if (sample_char =='\n'):
            break
        Y0 = np.zeros(shape=(1,num_decoder_tokens))
        Y0[0,sample_out_index]=1.
        prediction.append(sample_char)
    return prediction


actual, predicted = list(), list()
for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    actual_sentence = target_texts[seq_index: seq_index + 1][0]
    actual_sentence = actual_sentence[1:len(actual_sentence)-1]
    actual.append(actual_sentence)
    predicted.append(decoded_sentence)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)

print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

#BLEU-1: 0.570254
#BLEU-2: 0.755151
#BLEU-3: 0.844929
#BLEU-4: 0.868994
