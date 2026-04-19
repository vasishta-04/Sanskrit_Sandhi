# infer.py  –  Sandhi Vicceda inference (no retraining)
# --------------------------------------------------------
# Usage:
#   python infer.py                          # interactive mode
#   python infer.py रामायण                  # single word from CLI
#   python infer.py word1 word2 word3 ...   # batch from CLI
#
# Requirements: the 4 artefacts produced by training must exist
# in the SAME directory as this script:
#   bilstm.h5, bis2s.h5, window_metadata.json, split_metadata.json
# --------------------------------------------------------

import os, sys, json
import numpy as np

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
#  BUILD MODELS FROM SCRATCH + LOAD WEIGHTS
#  (avoids model_from_json / from_config entirely, so there is
#   no batch_shape / shape / batch_input_shape version conflict)
# =============================================================

def build_window_model(num_tokens: int, maxlen: int,
                       latent_dim: int = 64) -> keras.Model:
    """
    Recreates the BiLSTM window-detection model architecture.
    Matches predict_sandhi_window_bilstm.py exactly.
    """
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
    """
    Recreates the Seq2Seq split model and returns:
      (training_model, encoder_inference_model, decoder_inference_model)
    Matches split_sandhi_window_seq2seq_bilstm.py exactly.
    """
    # ── Training model (needed to restore weights) ────────────
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

    # ── Inference encoder ─────────────────────────────────────
    encoder_model = Model(encoder_inputs, encoder_states)

    # ── Inference decoder ─────────────────────────────────────
    dec_state_h_in = Input(shape=(latent_dim * 2,))
    dec_state_c_in = Input(shape=(latent_dim * 2,))

    dec_out, new_h, new_c = decoder_lstm(decoder_inputs,
                                          initial_state=[dec_state_h_in,
                                                         dec_state_c_in])
    dec_out = decoder_dense(dec_out)

    decoder_model = Model(
        [decoder_inputs, dec_state_h_in, dec_state_c_in],
        [dec_out, new_h, new_c]
    )

    return training_model, encoder_model, decoder_model


def load_weights_safe(model: keras.Model, filepath: str):
    """
    Load weights from an .h5 file into an already-built model.
    Tries keras load_weights first; falls back to manual h5 reading.
    """
    try:
        model.load_weights(filepath)
        return
    except Exception:
        pass

    # Manual fallback: read weight arrays directly from h5
    with h5py.File(filepath, "r") as f:
        wg = f.get("model_weights") or f
        for layer in model.layers:
            if layer.name not in wg:
                continue
            grp = wg[layer.name]
            wnames = grp.attrs.get("weight_names", [])
            weights = [grp[wn][()] for wn in wnames]
            if weights:
                try:
                    layer.set_weights(weights)
                except Exception as e:
                    print(f"  [skip layer '{layer.name}']: {e}")


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
        print("Loading metadata …")
        with open(self._path("window_metadata.json"), "r", encoding="utf-8") as f:
            meta_w = json.load(f)
        with open(self._path("split_metadata.json"), "r", encoding="utf-8") as f:
            meta_s = json.load(f)

        self.char2idx    = meta_w["char2idx"]
        self.maxlen_w    = meta_w["maxlen"]
        self.token_index = meta_s["token_index"]
        self.rev_token   = {i: c for c, i in self.token_index.items()}
        self.max_enc_len = meta_s["max_encoder_seq_length"]
        self.max_dec_len = meta_s["max_decoder_seq_length"]

        num_tokens_w = len(self.char2idx)
        num_tokens_s = len(self.token_index)

        print("Building window model and loading weights …")
        self.model_window = build_window_model(num_tokens_w, self.maxlen_w)
        load_weights_safe(self.model_window, self._path("bilstm.h5"))

        print("Building seq2seq model and loading weights …")
        training_model, self.encoder_model, self.decoder_model = \
            build_split_model(num_tokens_s)
        load_weights_safe(training_model, self._path("bis2s.h5"))

        print("Models ready.\n")

    def predict(self, devanagari_word: str) -> dict:
        slp1 = transliterate(devanagari_word,
                              sanscript.DEVANAGARI, sanscript.SLP1)

        # ── Step 1: Window prediction ─────────────────────────
        pad_val = self.char2idx.get("*", 0)
        x_w = [self.char2idx.get(c, pad_val) for c in slp1]
        x_w_pad = pad_sequences([x_w], maxlen=self.maxlen_w,
                                 padding="post", value=pad_val)

        window_preds = self.model_window.predict(
            x_w_pad, verbose=0).reshape(self.maxlen_w)

        max_sum, max_start = 0, 0
        # Determine the best 5-char window
        for i in range(max(1, len(slp1) - self.INWORDLEN + 1)):
            s = float(sum(window_preds[i: i + self.INWORDLEN]))
            if s > max_sum:
                max_sum, max_start = s, i

        # Extract context
        prefix = slp1[:max_start]
        junction = slp1[max_start: max_start + self.INWORDLEN]
        suffix = slp1[max_start + self.INWORDLEN:]

        # ── Step 2: Seq2Seq split ─────────────────────────────
        num_tokens = len(self.token_index)
        unk = self.token_index.get("*", 0)

        enc_data = np.zeros((1, self.max_enc_len, num_tokens), dtype="float32")
        for t, c in enumerate(junction):
            if t >= self.max_enc_len:
                break
            enc_data[0, t, self.token_index.get(c, unk)] = 1.0

        # This is the split version of the 5-char window (e.g., "mAy" -> "ma+ay")
        split_junction = self._decode(enc_data, num_tokens)
        split_junction = split_junction.replace("&", "").replace("$", "").strip()

        # ── Step 3: Reconstruction ────────────────────────────
        # Combine the untouched prefix + the processed junction + untouched suffix
        full_split_slp1 = f"{prefix}{split_junction}{suffix}"

        full_split_dev = transliterate(full_split_slp1,
                                       sanscript.SLP1, sanscript.DEVANAGARI)
        
        return {
            "input_devanagari": devanagari_word,
            "slp1_input":       slp1,
            "junction_window":  junction,
            "split_slp1":       full_split_slp1,
            "split_devanagari": full_split_dev,
        }

    def _decode(self, enc_data: np.ndarray, num_tokens: int) -> str:
        """Autoregressive greedy decode using inference encoder/decoder."""
        states = self.encoder_model.predict(enc_data, verbose=0)

        target_seq = np.zeros((1, 1, num_tokens), dtype="float32")
        target_seq[0, 0, self.token_index["&"]] = 1.0

        decoded = ""
        for _ in range(self.max_dec_len + 5):
            out, h, c = self.decoder_model.predict(
                [target_seq] + states, verbose=0)

            token_idx  = int(np.argmax(out[0, -1, :]))
            token_char = self.rev_token.get(token_idx, "")

            if token_char in ("$", ""):
                break
            decoded += token_char

            target_seq = np.zeros((1, 1, num_tokens), dtype="float32")
            target_seq[0, 0, token_idx] = 1.0
            states = [h, c]

        return decoded


# =============================================================
#  CLI entry-point
# =============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine = SandhiVicceda(model_dir=script_dir)

    words = sys.argv[1:]

    if words:
        for word in words:
            result = engine.predict(word)
            print(f"Input : {result['input_devanagari']}")
            print(f"Split : {result['split_devanagari']}  "
                  f"(SLP1: {result['split_slp1']})")
            print()
    else:
        print("Sandhi Vicceda – interactive mode")
        print("Type 'quit' or press Ctrl-C to exit.\n")
        while True:
            try:
                word = input("Enter Sanskrit compound (Devanagari): ").strip()
                if not word or word.lower() in ("quit", "exit", "q", "Bye"):
                    print("\nBye.. Bye..")
                    break
                result = engine.predict(word)
                print(f"  Split (Devanagari) : {result['split_devanagari']}")
                print(f"  Split (SLP1)       : {result['split_slp1']}")
                print(f"  Junction window    : {result['junction_window']}\n")
            except KeyboardInterrupt:
                print("\nBye.. Bye..")
                break


if __name__ == "__main__":
    main()