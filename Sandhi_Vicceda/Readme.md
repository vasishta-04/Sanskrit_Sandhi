# Sandhi Vicceda — Sanskrit Sandhi Splitter

Given a Sanskrit compound word in Devanagari, this module identifies the sandhi junction and recovers the two original words. It is the **inverse** of the Sandhi Combiner.

The pipeline is two-stage: a **window detection** model first locates the junction region inside the compound, and a **sequence-to-sequence splitter** model then decodes that region into the two constituent parts.

---

## Files

```
Sandhi_Vicceda/
├── splitting.py                          # Main inference script (batch + interactive)
├── sandhi_vicceda.py                     # Training orchestrator
├── predict_sandhi_window_bilstm.py       # Stage 1: window detection (training)
├── split_sandhi_window_seq2seq_bilstm.py # Stage 2: seq2seq splitter (training)
├── train_test_data_prepare.py            # Dataset parser & SLP1 vectoriser
├── devnagri_reader.py                    # Devanagari text cleaner
├── bilstm.h5                             # Trained window-detection model weights
├── bis2s.h5                              # Trained seq2seq splitter model weights
├── window_metadata.json                  # Char index & maxlen for window model
├── split_metadata.json                   # Token index & sequence lengths for splitter
├── input.txt                             # Example input file (compound words)
└── output.txt                            # Example output file
```

---

## Installation

```bash
pip install tensorflow tf-keras indic-transliteration numpy scikit-learn h5py
```

Python 3.10+ recommended.

---

## Usage

### Batch mode

Create an input file with one Devanagari compound word per line:

```
हितोपदेश
हिमालय
ज्वरान्त
```

Then run:

```bash
python splitting.py --file input.txt output.txt
```

Output format — one result per line:

```
हितोपदेश -> हित+उपदेश
हिमालय -> हिम्+आलय
ज्वरान्त -> ज्वर+अन्त
```

### Interactive mode

```bash
python splitting.py
# Mode: Interactive (Type 'q' to quit or use --file <in> <out>)
# Word: महोत्सव
# Result: मह+उत्सव
```

### Single word via command-line argument

```bash
python splitting.py हितोपदेश
# हितोपदेश -> हित+उपदेश
```

---

## How It Works

All processing is done in **SLP1** — a lossless ASCII transliteration of Sanskrit — to keep the model's character vocabulary small and consistent.

**Inference pipeline for `predict(devanagari_word)`:**

### Stage 1 — Window Detection (`bilstm.h5`)

1. Transliterate the input Devanagari word to SLP1.
2. Encode each character as an integer index (from `window_metadata.json`), pad to `maxlen = 50`.
3. Feed to the BiLSTM window model. It outputs a sigmoid activation for each character position, indicating the probability that this position belongs to the sandhi junction.
4. Slide a window of length 5 (`INWORDLEN`) across the character sequence and select the starting position with the highest summed activation — this is the predicted junction region.

### Stage 2 — Junction Splitting (`bis2s.h5`)

5. Extract the 5-character junction window from the SLP1 string.
6. Encode as a one-hot matrix and feed to the BiLSTM seq2seq splitter.
7. The decoder autoregressively generates the split junction (e.g. `"ra+A"` for a vowel sandhi).
8. Reconstruct the full SLP1 split: `prefix + split_junction + suffix`.
9. Transliterate back to Devanagari.

---

## Model Architecture

### Stage 1 — Window Model

A character-level **BiLSTM with embedding** that assigns a per-position probability to the sandhi window.

```
Input: padded integer character indices (maxlen = 50)
          │
    Embedding (vocab_size → 8 dims)
          │
    Bidirectional LSTM (latent_dim = 64, return_sequences=True)
          │
    Dropout (0.5)
          │
    Dense(1, activation='sigmoid')  ← one probability per character
          │
Output: [p₀, p₁, …, p₄₉]   (junction likelihood per position)
```

The window with the highest sum over 5 consecutive positions is the predicted junction.

### Stage 2 — Splitter Model

A character-level **seq2seq BiLSTM** that maps the 5-character junction window to the two sub-word endings separated by `+`.

```
Input: 5-char SLP1 junction (one-hot, max_encoder_seq_length = 5)
          │
    Bidirectional LSTM encoder (latent_dim = 128)
    → concatenated [forward_h + backward_h, forward_c + backward_c]
          │
    LSTM decoder (latent_dim × 2 = 256)
    → softmax over token vocabulary
          │
Output: "word1_end+word2_start"  in SLP1
        e.g. "it+up"  →  reconstructed as हित + उपदेश
```

Special tokens: `&` = start, `$` = stop, `*` = padding.

**Training hyperparameters:**

| Stage | Batch size | Epochs | Optimiser | Loss |
|---|---|---|---|---|
| Window detection | 64 | 40 | Adam | binary cross-entropy |
| Seq2seq splitter | 64 | 30 | RMSprop | categorical cross-entropy |

---

## Training

Run the full two-stage training pipeline from the `Sandhi_Vicceda/` directory:

```bash
python sandhi_vicceda.py
# Reads:  ../Data/sandhiset.txt
# Writes: bilstm.h5
#         bis2s.h5
#         window_metadata.json
#         split_metadata.json
```

**What `sandhi_vicceda.py` does:**

1. Calls `train_test_data_prepare.get_xy_data()` to parse `sandhiset.txt` into structured records.
2. Splits data 80/20 (train/test, `random_state=1`).
3. Calls `train_predict_sandhi_window()` to train the window model and predict junction starts on the test set.
4. Uses the predicted window positions to select 5-character junction windows for each test sample.
5. Calls `train_sandhi_split()` to train the seq2seq model and predict split junctions on the test set.
6. Reconstructs full split words and prints accuracy (passed / total).

### Training data format (`sandhiset.txt`)

```
<compound>  =>  <word1>  +  <word2>
```

The parser (`train_test_data_prepare.py`) extracts structured records:

```python
[slp1_word1_trimmed,    # suffix of word1 at junction
 slp1_word2_trimmed,    # prefix of word2 at junction
 slp1_junction,         # the 5-char window in the compound
 slp1_compound,         # full compound in SLP1
 start,                 # junction start index
 end,                   # junction end index  (end - start == 5)
 slp1_word1_full,       # full word1
 slp1_word2_full]       # full word2
```

Filtering criteria: `difflen` (len(compound) − len(word1) − len(word2)) must be in (−2, +1), compound length ≤ 50 and ≥ 5.

---

## Metadata Files

### `window_metadata.json`

| Key | Description |
|---|---|
| `char2idx` | Maps each SLP1 character (+ `*` padding) to an integer index |
| `maxlen` | Maximum compound length seen during training (50) |

### `split_metadata.json`

| Key | Description |
|---|---|
| `token_index` | Maps each SLP1 character + special tokens to an integer index |
| `max_encoder_seq_length` | Maximum junction input length (5) |
| `max_decoder_seq_length` | Maximum decoder output length (10) |

---

## Example Results

| Input (compound) | Predicted split |
|---|---|
| हितोपदेश | हित+उपदेश |
| हिमालय | हिम्+आलय |
| ज्वरान्त | ज्वर+अन्त |
| रोगमार्ग | रोग+मार्ग |
| देहबल | देह+बल |
| रामायण | राम्+आयण |
