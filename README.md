# Sanskrit Sandhi Processing — Project Overview

A deep-learning pipeline for two complementary Sanskrit linguistic tasks:

1. **Sandhi (Sandhi Combiner)** — given two Sanskrit words, predict the phonetically merged compound word.
2. **Sandhi Vicceda (Sandhi Splitter)** — given a Sanskrit compound word, predict where it splits and recover the two original words.

Both models operate on Devanagari input, transliterate internally to SLP1 encoding, apply neural sequence-to-sequence inference, and return Devanagari output.

---

## Repository Layout

```
project-root/
├── Data/                          # Training & evaluation corpora
│   ├── sandhiset.txt              # Primary sandhi dataset (~12 MB, main training source)
│   ├── sanskrit_sandhi_4000.txt   # Supplementary 4000-pair sandhi set
│   └── SandhiKosh/
│       ├── Astaadhyaayii.txt      # Pāṇini's Ashtadhyayi (sandhi evaluation)
│       ├── Bhagvad_Gita.txt       # Bhagavad Gita (sandhi evaluation)
│       └── literature.txt         # General Sanskrit literature (sandhi evaluation)
│
├── Sandhi/                        # Sandhi combiner module
│   ├── combine.py                 # Inference entry-point (batch + interactive)
│   ├── single_dict_seq2seq_bilstm_sandhi.py  # Training script
│   ├── sandhi_data_prepare.py     # Dataset parsing & vectorisation
│   ├── devnagri_reader.py         # Devanagari text cleaner (shared utility)
│   ├── bis2s.h5                   # Trained BiLSTM seq2seq weights
│   ├── combine_metadata.json      # Token index & sequence-length constants
│   ├── input.txt                  # Example input (word pairs, one per line)
│   └── output.txt                 # Example output from combine.py
│
└── Sandhi_Vicceda/                # Sandhi splitter module
    ├── splitting.py               # Inference entry-point (batch + interactive)
    ├── sandhi_vicceda.py          # Training orchestrator
    ├── predict_sandhi_window_bilstm.py   # Stage 1: window detection training
    ├── split_sandhi_window_seq2seq_bilstm.py  # Stage 2: splitting training
    ├── train_test_data_prepare.py # Dataset parsing & vectorisation
    ├── devnagri_reader.py         # Devanagari text cleaner (shared utility)
    ├── bilstm.h5                  # Trained window-detection model weights
    ├── bis2s.h5                   # Trained seq2seq splitter model weights
    ├── window_metadata.json       # Char index & max-length for window model
    ├── split_metadata.json        # Token index & sequence lengths for split model
    ├── input.txt                  # Example input (compound words, one per line)
    └── output.txt                 # Example output from splitting.py
```

---

## How the Two Tasks Relate

```
Word 1 + Word 2  ──[Sandhi Combiner]──►  Compound Word
                                               │
                                     [Sandhi Vicceda]
                                               │
                                               ▼
                                     Word 1  +  Word 2
```

The two modules are **inverses of each other**. They share the same training corpus (`sandhiset.txt`), the same SLP1 transliteration scheme, and the same Devanagari text cleaner (`devnagri_reader.py`).

---

## Installation

```bash
pip install tensorflow tf-keras indic-transliteration scikit-learn numpy h5py
```

Python 3.10+ is recommended. Both modules set `TF_USE_LEGACY_KERAS=1` internally.

---

## Data

The `sandhiset.txt` file is the canonical training dataset. Each line follows the format:

```
<compound>  =>  <word1>  +  <word2>
```

Example:
```
रामायण  =>  राम  +  अयन
```

The `SandhiKosh/` sub-folder contains three classical texts used as held-out evaluation sets for the Sandhi Combiner.

---

## Quick Start

### Sandhi Combiner — combine two words

```bash
# Batch mode
python Sandhi/combine.py Sandhi/input.txt Sandhi/output.txt

# Interactive mode
python Sandhi/combine.py
# word1, word2 : राम,अयन
# → राम + अयन = रामयन
```

### Sandhi Vicceda — split a compound word

```bash
# Batch mode
python Sandhi_Vicceda/splitting.py --file Sandhi_Vicceda/input.txt Sandhi_Vicceda/output.txt

# Interactive mode
python Sandhi_Vicceda/splitting.py
# Word: हितोपदेश
# Result: हित+उपदेश

# Single word via argument
python Sandhi_Vicceda/splitting.py हितोपदेश
```

---

## Model Architecture Summary

| Component | Module | Architecture |
|---|---|---|
| Sandhi Combiner | `Sandhi/` | BiLSTM encoder → LSTM decoder (seq2seq) |
| Window Detector | `Sandhi_Vicceda/` | BiLSTM + Embedding + Dropout, sigmoid output per character |
| Sandhi Splitter | `Sandhi_Vicceda/` | BiLSTM encoder → LSTM decoder (seq2seq) |

All models use SLP1 one-hot or index encoding over a ~50-character Sanskrit phoneme alphabet.

---

## Training

Retrain the Sandhi Combiner:
```bash
cd Sandhi
python single_dict_seq2seq_bilstm_sandhi.py
# Reads: ../Data/sandhiset.txt
# Writes: bis2s.h5, combine_metadata.json
```

Retrain the Sandhi Vicceda pipeline:
```bash
cd Sandhi_Vicceda
python sandhi_vicceda.py
# Reads: ../Data/sandhiset.txt
# Writes: bilstm.h5, bis2s.h5, window_metadata.json, split_metadata.json
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` / `tf-keras` | Model training & inference |
| `indic-transliteration` | Devanagari ↔ SLP1 conversion |
| `scikit-learn` | Train/test split |
| `numpy` | Array operations |
| `h5py` | Safe weight loading from `.h5` files |
