import os, sys, json
import numpy as np

# Suppress TensorFlow logs for a cleaner CLI experience
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tf_keras as keras
from tf_keras.layers import (Input, LSTM, Dense, Bidirectional, Embedding,
                             Dropout, Concatenate)
from tf_keras.models import Model
from tf_keras.preprocessing.sequence import pad_sequences

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import h5py

# =============================================================
#  BUILD MODELS FROM SCRATCH
# =============================================================

def build_window_model(num_tokens: int, maxlen: int,
                       latent_dim: int = 64) -> keras.Model:
    inputword = Input(shape=(maxlen,))
    embed = Embedding(input_dim=num_tokens, output_dim=8,
                      input_length=maxlen, mask_zero=True)(inputword)
    bilstm = Bidirectional(LSTM(latent_dim, return_sequences=True,
                                return_state=True))
    out, *_ = bilstm(embed)
    outd = Dropout(0.5)(out)
    output = Dense(1, activation="sigmoid")(outd)
    return Model(inputword, output)

def build_split_model(num_tokens: int, latent_dim: int = 128):
    encoder_inputs = Input(shape=(None, num_tokens))
    encoder = Bidirectional(LSTM(latent_dim, return_state=True, dropout=0.5))
    enc_out, fh, fc, bh, bc = encoder(encoder_inputs)
    state_h = Concatenate()([fh, bh])
    state_c = Concatenate()([fc, bc])
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_tokens))
    decoder_lstm = LSTM(latent_dim * 2, return_sequences=True,
                        return_state=True, dropout=0.5)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(encoder_inputs, encoder_states)
    dec_state_h_in = Input(shape=(latent_dim * 2,))
    dec_state_c_in = Input(shape=(latent_dim * 2,))
    dec_out, new_h, new_c = decoder_lstm(decoder_inputs,
                                          initial_state=[dec_state_h_in,
                                                         dec_state_c_in])
    dec_out = decoder_dense(dec_out)
    decoder_model = Model([decoder_inputs, dec_state_h_in, dec_state_c_in],
                          [dec_out, new_h, new_c])

    return training_model, encoder_model, decoder_model

def load_weights_safe(model: keras.Model, filepath: str):
    try:
        model.load_weights(filepath)
        return
    except Exception:
        with h5py.File(filepath, "r") as f:
            wg = f.get("model_weights") or f
            for layer in model.layers:
                if layer.name not in wg: continue
                grp = wg[layer.name]
                wnames = grp.attrs.get("weight_names", [])
                weights = [grp[wn][()] for wn in wnames]
                if weights:
                    try: layer.set_weights(weights)
                    except: pass

# =============================================================
#  INFERENCE ENGINE
# =============================================================

class SandhiVicceda:
    INWORDLEN = 5

    def __init__(self, model_dir: str = "."):
        self.model_dir = model_dir
        self._load()

    def _path(self, name):
        return os.path.join(self.model_dir, name)

    def _load(self):
        print("Loading metadata...")
        with open(self._path("window_metadata.json"), "r", encoding="utf-8") as f:
            meta_w = json.load(f)
        with open(self._path("split_metadata.json"), "r", encoding="utf-8") as f:
            meta_s = json.load(f)

        self.char2idx = meta_w["char2idx"]
        self.maxlen_w = meta_w["maxlen"]
        self.token_index = meta_s["token_index"]
        self.rev_token = {i: c for c, i in self.token_index.items()}
        self.max_enc_len = meta_s["max_encoder_seq_length"]
        self.max_dec_len = meta_s["max_decoder_seq_length"]

        num_tokens_w, num_tokens_s = len(self.char2idx), len(self.token_index)

        self.model_window = build_window_model(num_tokens_w, self.maxlen_w)
        load_weights_safe(self.model_window, self._path("bilstm.h5"))

        tm, self.encoder_model, self.decoder_model = build_split_model(num_tokens_s)
        load_weights_safe(tm, self._path("bis2s.h5"))
        print("Models ready.\n")

    def predict(self, devanagari_word: str) -> dict:
        slp1 = transliterate(devanagari_word, sanscript.DEVANAGARI, sanscript.SLP1)
        pad_val = self.char2idx.get("*", 0)
        x_w = [self.char2idx.get(c, pad_val) for c in slp1]
        x_w_pad = pad_sequences([x_w], maxlen=self.maxlen_w, padding="post", value=pad_val)
        
        window_preds = self.model_window.predict(x_w_pad, verbose=0).reshape(self.maxlen_w)
        max_sum, max_start = 0, 0
        for i in range(max(1, len(slp1) - self.INWORDLEN + 1)):
            s = float(sum(window_preds[i: i + self.INWORDLEN]))
            if s > max_sum: max_sum, max_start = s, i

        prefix, junction, suffix = slp1[:max_start], slp1[max_start: max_start+5], slp1[max_start+5:]
        
        num_tokens = len(self.token_index)
        unk = self.token_index.get("*", 0)
        enc_data = np.zeros((1, self.max_enc_len, num_tokens), dtype="float32")
        for t, c in enumerate(junction):
            if t < self.max_enc_len: enc_data[0, t, self.token_index.get(c, unk)] = 1.0

        split_j = self._decode(enc_data, num_tokens).replace("&", "").replace("$", "").strip()
        full_slp1 = f"{prefix}{split_j}{suffix}"
        
        return {
            "input": devanagari_word,
            "split": transliterate(full_slp1, sanscript.SLP1, sanscript.DEVANAGARI),
            "slp1": full_slp1
        }

    def _decode(self, enc_data: np.ndarray, num_tokens: int) -> str:
        states = self.encoder_model.predict(enc_data, verbose=0)
        target_seq = np.zeros((1, 1, num_tokens), dtype="float32")
        target_seq[0, 0, self.token_index["&"]] = 1.0
        decoded = ""
        for _ in range(self.max_dec_len + 5):
            out, h, c = self.decoder_model.predict([target_seq] + states, verbose=0)
            idx = int(np.argmax(out[0, -1, :]))
            char = self.rev_token.get(idx, "")
            if char in ("$", ""): break
            decoded += char
            target_seq = np.zeros((1, 1, num_tokens), dtype="float32")
            target_seq[0, 0, idx] = 1.0
            states = [h, c]
        return decoded

# =============================================================
#  FILE AND CLI LOGIC
# =============================================================

def run_file_mode(engine, in_file, out_file):
    if not os.path.exists(in_file):
        print(f"File not found: {in_file}")
        return
    
    with open(in_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(lines)} words...")
    with open(out_file, 'w', encoding='utf-8') as f:
        for word in lines:
            try:
                res = engine.predict(word)
                f.write(f"{res['input']} -> {res['split']}\n")
            except Exception as e:
                f.write(f"{word} -> ERROR ({e})\n")
    print(f"Done! Results saved to {out_file}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine = SandhiVicceda(model_dir=script_dir)

    if len(sys.argv) == 4 and sys.argv[1] == "--file":
        run_file_mode(engine, sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 1:
        for word in sys.argv[1:]:
            res = engine.predict(word)
            print(f"{res['input']} -> {res['split']}")
    else:
        print("Mode: Interactive (Type 'q' to quit or use --file <in> <out>)")
        while True:
            word = input("Word: ").strip()
            if word.lower() in ('q', 'quit', 'exit'): break
            if word: 
                res = engine.predict(word)
                print(f"Result: {res['split']}\n")

if __name__ == "__main__":
    main()