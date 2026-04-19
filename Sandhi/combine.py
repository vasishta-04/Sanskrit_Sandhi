# combine.py  –  Sandhi Combiner (inference only, no retraining)
# ---------------------------------------------------------------
# Usage:
#   python combine.py input.txt output.txt
#   Input file format: one pair per line →  word1,word2
#   Both words must be in Devanagari script.
#
# Requirements (same folder as this script):
#   bis2s.h5  +  combine_metadata.json
# ---------------------------------------------------------------

import os, sys, json
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tf_keras as keras
from tf_keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
from tf_keras.models import Model

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ── SLP1 character allowlist (same as sandhi_data_prepare.py) ──
_SWARAS    = list('aAiIuUeEoOfFxX')
_VYANJANAS = list('kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsshL|')
_OTHERS    = ['H', 'Z', 'V', 'M', '~', '/', '\\', '^', "'"]
_SLP1_CHARS = set(_SWARAS + _VYANJANAS + _OTHERS)

def _clean_slp1(text: str) -> str:
    return ''.join(c for c in text if c in _SLP1_CHARS)

# ── Training constants (must match single_dict_seq2seq_bilstm_sandhi.py) ──
FWL = 4   # last N chars of word1 fed to model
SWL = 2   # first N chars of word2 fed to model
LATENT_DIM = 16


class SandhiCombiner:
    """
    Loads a trained bis2s model once and exposes a combine() method.

    KEY INSIGHT (the bug fix):
    The training script trims word1 and word2 before feeding them to the model:
      - word1  →  last FWL (4) characters           prefix = word1[:-FWL]
      - word2  →  first SWL (2) characters          suffix = word2[SWL:]
      - output →  corresponding middle portion only

    So the model predicts only the JUNCTION portion.
    To get the full combined word we must reconstruct:
        full_output = prefix + model_junction + suffix
    """

    def __init__(self, model_path: str = "bis2s.h5",
                 meta_path: str = "combine_metadata.json"):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.token_index  = meta["token_index"]
        self.rev_token    = {i: c for c, i in self.token_index.items()}
        self.max_enc_len  = meta["max_encoder_seq_length"]
        self.max_dec_len  = meta["max_decoder_seq_length"]
        self.num_tokens   = len(self.token_index)

        self._build_and_load(model_path)
        print("Model loaded successfully.\n")

    # ── model construction ───────────────────────────────────────

    def _build_and_load(self, model_path: str):
        n = self.num_tokens

        # Training architecture (needed so weights map correctly)
        encoder_inputs = Input(shape=(None, n))
        encoder = Bidirectional(LSTM(LATENT_DIM, return_state=True))
        _, fh, fc, bh, bc = encoder(encoder_inputs)
        state_h = Concatenate()([fh, bh])
        state_c = Concatenate()([fc, bc])

        decoder_inputs = Input(shape=(None, n))
        decoder_lstm  = LSTM(LATENT_DIM * 2, return_sequences=True,
                             return_state=True)
        dec_out, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=[state_h, state_c])
        decoder_dense = Dense(n, activation="softmax")
        dec_out = decoder_dense(dec_out)

        training_model = Model([encoder_inputs, decoder_inputs], dec_out)
        training_model.load_weights(model_path)

        # Inference encoder
        self.encoder_model = Model(encoder_inputs, [state_h, state_c])

        # Inference decoder (reuses trained layers)
        dsh = Input(shape=(LATENT_DIM * 2,))
        dsc = Input(shape=(LATENT_DIM * 2,))
        dec_out2, new_h, new_c = decoder_lstm(decoder_inputs,
                                               initial_state=[dsh, dsc])
        dec_out2 = decoder_dense(dec_out2)
        self.decoder_model = Model([decoder_inputs, dsh, dsc],
                                   [dec_out2, new_h, new_c])

    # ── internal helpers ────────────────────────────────────────

    def _one_hot(self, text: str) -> np.ndarray:
        """Encode a SLP1 string as a one-hot matrix (1, max_enc_len, num_tokens)."""
        pad_idx = self.token_index.get('*', 0)
        x = np.zeros((1, self.max_enc_len, self.num_tokens), dtype="float32")
        for t, ch in enumerate(text):
            if t >= self.max_enc_len:
                break
            if ch in self.token_index:
                x[0, t, self.token_index[ch]] = 1.0
        # pad remaining positions
        for t in range(len(text), self.max_enc_len):
            x[0, t, pad_idx] = 1.0
        return x

    def _decode_junction(self, enc_data: np.ndarray) -> str:
        """Run the seq2seq decoder and return the raw SLP1 junction string."""
        states = self.encoder_model.predict(enc_data, verbose=0)

        target_seq = np.zeros((1, 1, self.num_tokens), dtype="float32")
        target_seq[0, 0, self.token_index['&']] = 1.0

        decoded = ""
        for _ in range(self.max_dec_len + 10):
            out, h, c = self.decoder_model.predict(
                [target_seq] + states, verbose=0)
            idx  = int(np.argmax(out[0, -1, :]))
            char = self.rev_token.get(idx, "")

            if char in ("$", "*", ""):
                break
            decoded += char

            target_seq = np.zeros((1, 1, self.num_tokens), dtype="float32")
            target_seq[0, 0, idx] = 1.0
            states = [h, c]

        return decoded

    # ── public API ──────────────────────────────────────────────

    def combine(self, word1: str, word2: str) -> str:
        """
        Combine two Sanskrit words (Devanagari) applying sandhi rules.

        Steps
        -----
        1. Transliterate both words to SLP1.
        2. Trim exactly as the training script did:
               model_w1  = last FWL (4) chars of word1_slp1
               model_w2  = first SWL (2) chars of word2_slp1
               prefix    = word1_slp1 up to the trimmed part
               suffix    = word2_slp1 after the trimmed part
        3. Feed  "model_w1 + '+' + model_w2"  to the seq2seq model.
        4. Reconstruct full answer:
               full_slp1 = prefix + junction + suffix
        5. Transliterate back to Devanagari.
        """
        # Step 1 – SLP1 conversion
        w1_slp1 = _clean_slp1(
            transliterate(word1, sanscript.DEVANAGARI, sanscript.SLP1))
        w2_slp1 = _clean_slp1(
            transliterate(word2, sanscript.DEVANAGARI, sanscript.SLP1))

        # Step 2 – Trim exactly like training
        if len(w1_slp1) > FWL:
            prefix   = w1_slp1[:-FWL]   # everything BEFORE the last FWL chars
            model_w1 = w1_slp1[-FWL:]   # last FWL chars go to model
        else:
            prefix   = ""
            model_w1 = w1_slp1

        if len(w2_slp1) > SWL:
            suffix   = w2_slp1[SWL:]    # everything AFTER the first SWL chars
            model_w2 = w2_slp1[:SWL]    # first SWL chars go to model
        else:
            suffix   = ""
            model_w2 = w2_slp1

        # Step 3 – Encode and predict junction
        model_input = f"{model_w1}+{model_w2}"
        enc_data    = self._one_hot(model_input)
        junction    = self._decode_junction(enc_data)

        # Step 4 – Reconstruct full SLP1 word
        full_slp1 = prefix + junction + suffix

        # Step 5 – Back to Devanagari
        return transliterate(full_slp1, sanscript.SLP1, sanscript.DEVANAGARI)


# =============================================================
#  File processing
# =============================================================

def process_file(input_path: str, output_path: str,
                 combiner: SandhiCombiner):
    """
    Read word pairs from input_path (one 'word1,word2' per line)
    and write results to output_path.
    """
    if not os.path.exists(input_path):
        print(f"Error: input file '{input_path}' not found.")
        return

    with open(input_path,  "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        print(f"Processing '{input_path}' …")
        ok = skip = 0

        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 2:
                print(f"  [line {lineno}] Skipping – expected 'word1,word2', got: {line!r}")
                skip += 1
                continue

            w1, w2 = parts[0].strip(), parts[1].strip()
            try:
                combined = combiner.combine(w1, w2)
                fout.write(f"{w1} + {w2} = {combined}\n")
                ok += 1
            except Exception as e:
                print(f"  [line {lineno}] Error for '{w1},{w2}': {e}")
                skip += 1

    print(f"Done. {ok} processed, {skip} skipped. Results → '{output_path}'")


# =============================================================
#  CLI
# =============================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "bis2s.h5")
    meta_path  = os.path.join(script_dir, "combine_metadata.json")

    combiner = SandhiCombiner(model_path=model_path, meta_path=meta_path)

    if len(sys.argv) == 3:
        process_file(sys.argv[1], sys.argv[2], combiner)

    elif len(sys.argv) == 1:
        # Interactive mode
        print("Sandhi Combiner – interactive mode")
        print("Type two Devanagari words separated by comma, or 'quit' to exit.\n")
        while True:
            try:
                line = input("word1, word2 : ").strip()
                if not line or line.lower() in ("quit", "exit", "q"):
                    break
                parts = line.split(',')
                if len(parts) != 2:
                    print("  Please enter exactly two words separated by a comma.")
                    continue
                w1, w2 = parts[0].strip(), parts[1].strip()
                result = combiner.combine(w1, w2)
                print(f"  {w1} + {w2} = {result}\n")
            except KeyboardInterrupt:
                print("\nBye!")
                break
    else:
        print("Usage:")
        print("  python combine.py                        # interactive mode")
        print("  python combine.py input.txt output.txt   # batch mode")


if __name__ == "__main__":
    main()