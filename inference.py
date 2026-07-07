import tensorflow as tf
import json
import numpy as np
import os
import string
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ==========================================
# CONFIGURATION
# ==========================================
# Put the name of your weights file here (e.g., 'seq2seq_attention.weights.h5' or 'seq2seq_weights_cuDNN.weights.h5')
WEIGHTS_PATH = "seq2seq_attention.weights.h5" 

# Set to True if testing the Attention model, False for the Vanilla/cuDNN model
USE_ATTENTION = True

# Vocabulary limits vary based on the model trained
VOCAB_SIZE = 20000 if USE_ATTENTION else 10000
MAX_LENGTH = 50 if USE_ATTENTION else 30

embed_size = 256
hidden_size = 512
# ==========================================

print("Loading vocabularies...")
with open('eng_vocab.json', 'r', encoding='utf-8') as f:
    eng_vocab = json.load(f)
with open('hind_vocab.json', 'r', encoding='utf-8') as f:
    hind_vocab = json.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# === ENCODER ===
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        # return_sequences=True is required for Attention, harmless for Vanilla
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)

    def call(self, x):
        embedded = self.embedding(x)
        enc_output, hidden, cell = self.lstm(embedded)
        return enc_output, hidden, cell

# === VANILLA DECODER ===
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, cell):
        embedded = self.embedding(x)
        lstm_output, hidden, cell = self.lstm(embedded, initial_state=[hidden, cell])
        output = self.dense(lstm_output)
        return output, hidden, cell

# === ATTENTION DECODER ===
class DecoderWithAttention(tf.keras.Model):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(DecoderWithAttention, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_size)
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=True)
        self.attention = tf.keras.layers.Attention()
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden, cell, enc_output):
        embedded = self.embedding(x)
        dec_output, h, c = self.lstm(embedded, initial_state=[hidden, cell])
        context_vector = self.attention([dec_output, enc_output])
        combined = self.concat([dec_output, context_vector])
        output = self.dense(combined)
        return output, h, c

# === SEQ2SEQ WRAPPER ===
class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder, decoder, tgt_vocab_size, use_attention=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_vocab_size = tgt_vocab_size
        self.use_attention = use_attention

    def call(self, inputs, training=False):
        src, tgt = inputs[0], inputs[1]
        enc_output, hidden, cell = self.encoder(src)
        
        batch_size = tf.shape(src)[0]
        tgt_len = tf.shape(tgt)[1] 
        
        i0 = tf.constant(1)
        input_token0 = tgt[:, 0:1] 
        all_outputs0 = tf.TensorArray(tf.float32, size=tgt_len - 1)

        def cond(i, input_token, h, c, all_outputs):
            return i < tgt_len
            
        def body(i, input_token, h, c, all_outputs):
            if self.use_attention:
                output, h, c = self.decoder(input_token, h, c, enc_output)
            else:
                output, h, c = self.decoder(input_token, h, c)
                
            all_outputs = all_outputs.write(i - 1, output)
            next_input = tf.argmax(output, axis=-1, output_type=tf.int32)
            return i + 1, next_input, h, c, all_outputs

        _, _, _, _, final_outputs = tf.while_loop(
            cond, body,
            loop_vars=[i0, input_token0, hidden, cell, all_outputs0]
        )

        outputs_stacked = final_outputs.stack()
        outputs_stacked = tf.squeeze(outputs_stacked, axis=2) 
        return tf.transpose(outputs_stacked, perm=[1, 0, 2]) 

# === INITIALIZE MODEL ===
encoder = Encoder(VOCAB_SIZE, embed_size, hidden_size)

if USE_ATTENTION:
    decoder = DecoderWithAttention(VOCAB_SIZE, embed_size, hidden_size)
else:
    decoder = Decoder(VOCAB_SIZE, embed_size, hidden_size)

model = Seq2Seq(encoder, decoder, tgt_vocab_size=VOCAB_SIZE, use_attention=USE_ATTENTION)

dummy_src = tf.zeros((1, 5), dtype=tf.int32)
dummy_tgt = tf.zeros((1, 5), dtype=tf.int32)
_ = model([dummy_src, dummy_tgt], training=False)

if WEIGHTS_PATH:
    try:
        model.load_weights(WEIGHTS_PATH)
        print(f"Model weights loaded successfully from {WEIGHTS_PATH}.")
    except Exception as e:
        print(f"Warning: Could not load weights from {WEIGHTS_PATH}.\nError: {e}")
else:
    print("Warning: WEIGHTS_PATH is vacant. Running with random initialization.")

def translate(sentence, max_len=MAX_LENGTH):
    rev_hind = {number: word for word, number in hind_vocab.items()}
    
    if USE_ATTENTION:
        sentence = clean_text(sentence)
        
    tokens = [eng_vocab.get('<SOS>', 1)]
    unk_id = eng_vocab.get('<UNK>', 3)
    tokens += [eng_vocab.get(word, unk_id) for word in sentence.split()]
    tokens += [eng_vocab.get('<EOS>', 2)]
    
    src_tensor = tf.convert_to_tensor([tokens], dtype=tf.int32)
    enc_output, hidden, cell = model.encoder(src_tensor)
    
    input_token = tf.convert_to_tensor([[hind_vocab.get('<SOS>', 1)]], dtype=tf.int32)
    translated_words = []
    
    for _ in range(max_len):
        if USE_ATTENTION:
            output, hidden, cell = model.decoder(input_token, hidden, cell, enc_output)
        else:
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            
        input_token = tf.argmax(output, axis=-1, output_type=tf.int32)
        pred_index = input_token.numpy()[0, 0]
        
        if pred_index == hind_vocab.get('<EOS>', 2):
            break
            
        pred_word = rev_hind.get(pred_index, '<UNK>')
        translated_words.append(pred_word)
        
    return " ".join(translated_words)

if __name__ == "__main__":
    print("\n--- Testing Predefined Sentences ---")
    test_sentences = [
        "how are you",
        "what is your name",
        "where are you going",
        "i love you",
        "this is a good book",
        "he is a good boy",
        "please give me some water",
        "i am going home"
    ]

    for sent in test_sentences:
        hindi_out = translate(sent)
        print(f"English: {sent}")
        print(f"Hindi  : {hindi_out}")
        print("-" * 30)

    print("\n--- Interactive Translation ---")
    print("Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            text = input("English: ")
            if text.strip().lower() in ['quit', 'exit']:
                break
            hindi_out = translate(text)
            print(f"Hindi: {hindi_out}\n")
        except KeyboardInterrupt:
            break
