# Sandhi — Sanskrit Sandhi Combiner

Given two Sanskrit words in Devanagari, this module predicts the phonetically merged compound word produced by applying Sanskrit sandhi rules. Inference uses a pre-trained BiLSTM sequence-to-sequence model and requires no retraining.

---

## Files

```
Sandhi/
├── combine.py                          # Main inference script (batch + interactive)
├── single_dict_seq2seq_bilstm_sandhi.py # Training script
├── sandhi_data_prepare.py              # Dataset parser & SLP1 vectoriser
├── devnagri_reader.py                  # Devanagari text cleaner
├── bis2s.h5                            # Trained model weights
├── combine_metadata.json               # Token index & sequence-length constants
├── input.txt                           # Example input file
└── output.txt                          # Example output file
```

---

## Installation

```bash
pip install tensorflow tf-keras indic-transliteration numpy scikit-learn
```

Python 3.10+ recommended.

---

## Usage

### Batch mode

Create an input file with one comma-separated word pair per line (Devanagari):

```
राम,अयन
हित,उपदेश
हिम,आलय
```

Then run:

```bash
python combine.py input.txt output.txt
```

Output format — one result per line:

```
राम + अयन = रामयन
हित + उपदेश = हितोपदेश
हिम + आलय = हिमालय
```

### Interactive mode

```bash
python combine.py
# Sandhi Combiner – interactive mode
# word1, word2 : शिव,आलय
#   शिव + आलय = शिवालय
```

Type `quit` or `exit` to stop.

---

## How It Works

The model operates entirely in **SLP1** — a lossless ASCII transliteration of Sanskrit — to avoid Unicode complexity during training.

**Inference pipeline for `combine(word1, word2)`:**

1. Transliterate both Devanagari inputs to SLP1 using `indic-transliteration`.
2. Trim the inputs to the junction region that governs sandhi:
   - Take the **last 4 characters** of `word1` (FWL = 4).
   - Take the **first 2 characters** of `word2` (SWL = 2).
   - Remember the untrimmed prefix of `word1` and suffix of `word2`.
3. Concatenate as `"<word1_tail>+<word2_head>"` and encode as a one-hot matrix.
4. Feed to the BiLSTM encoder → LSTM decoder to predict the junction characters.
5. Reconstruct the full word: `prefix + junction + suffix`.
6. Transliterate the SLP1 result back to Devanagari.

**Why trim?** Sandhi only affects the phonemes at the boundary between two words. Training and inference on only the junction region keeps the model small and accurate, while the untrimmed parts are passed through unchanged.

---

## Model Architecture

The model is a character-level **sequence-to-sequence BiLSTM** trained to map the junction input to the merged junction output.

```
Input: "mAna+su"  (SLP1 junction characters joined with '+')
         │
    ┌────▼─────────────────────────────────┐
    │  Bidirectional LSTM encoder           │
    │  latent_dim = 16                      │
    │  → [forward_h, forward_c,             │
    │     backward_h, backward_c]           │
    └────────────┬──────────────────────────┘
                 │  concatenated states
    ┌────────────▼──────────────────────────┐
    │  LSTM decoder  (latent_dim × 2 = 32)  │
    │  → softmax over token vocabulary      │
    └────────────┬──────────────────────────┘
                 │
Output: "mAnaso"  (predicted junction in SLP1)
```

Decoder uses `&` as the start token and `$` as the stop token. `*` is the padding symbol.

**Training hyperparameters:**
- Batch size: 64
- Epochs: 100
- Optimiser: RMSprop
- Loss: categorical cross-entropy

---

## Training

The training script reads from the shared corpus and saves weights + metadata:

```bash
python single_dict_seq2seq_bilstm_sandhi.py
# Reads:  ../Data/sandhiset.txt
# Writes: bis2s.h5
#         combine_metadata.json
```

After training, the script evaluates accuracy on:
- The held-out test split (80/20 split with `random_state=1`)
- Three classical text corpora: `Astaadhyaayii.txt`, `Bhagvad_Gita.txt`, `literature.txt`

### Data format (`sandhiset.txt`)

```
<compound>  =>  <word1>  +  <word2>
```

The parser (`sandhi_data_prepare.py`) filters pairs where the sandhi transformation is a short, learnable boundary change (`difflen` between −2 and +1).

---

## Metadata (`combine_metadata.json`)

| Key | Description |
|---|---|
| `token_index` | Maps each SLP1 character + special tokens to an integer index |
| `max_encoder_seq_length` | Maximum input sequence length (7) |
| `max_decoder_seq_length` | Maximum output sequence length (9) |
| `characters` | Sorted list of all characters seen during training |

---

## Example Results

| word1 | word2 | Combined |
|---|---|---|
| राम | अयन | रामयन |
| शिव | आलय | शिवालय |
| हित | उपदेश | हितोपदेश |
| हिम | आलय | हिमालय |
| ज्वर | अन्त | ज्वरान्त |
