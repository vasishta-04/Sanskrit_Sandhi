# Neural Sanskrit Sandhi Viccheda Engine

## Project Overview
This project implements a sophisticated two-stage deep learning pipeline to solve the problem of **Sanskrit Sandhi Viccheda** (phonetic de-compounding). By combining character-level Bi-directional LSTMs (BiLSTM) for junction detection and Sequence-to-Sequence (Seq2Seq) models for splitting, the engine effectively breaks complex Sanskrit compounds into their original constituent words.

The system is trained on an extensive corpus of over 160,000 samples, utilizing classical texts such as the **Bhagavad Gita**, **Astaadhyaayii**, and various Sanskrit literature datasets.

## Technical Architecture

The engine moves away from traditional rule-based grammar and uses a neural approach:

1.  **Preprocessing & Transliteration:** * Normalizes raw Devanagari text using `devnagri_reader.py`.
    * Converts text into the **SLP1 (Sanskrit Library Phonetic)** scheme using `indic-transliteration` to streamline feature learning for the neural network.
2.  **Junction Prediction (Window Model):** * A **Bi-directional LSTM** identifies the exact 5-character "junction" where the Sandhi occurs.
    * This reduces the problem space from the entire word to a specific phonetic window.
3.  **Neural Splitter (Seq2Seq Model):** * An **Encoder-Decoder BiLSTM** maps the 5-character junction window back into the "Word1 + Word2" components.
    * It utilizes start (`&`) and end (`$`) tokens for accurate sequence generation.

## Repository Structure

* `splitting.py`: The primary inference script for end-to-end prediction (CLI and File mode).
* `predict_sandhi_window_bilstm.py`: Logic for the Junction Detection BiLSTM model.
* `split_sandhi_window_seq2seq_bilstm.py`: Logic for the Seq2Seq Splitting model.
* `sandhi_vicceda.py`: High-level script for training the combined pipeline.
* `train_test_data_prepare.py`: Dataset generator that handles windowing and SLP1 conversion.
* `devnagri_reader.py`: Text cleaning utility to remove non-Sanskrit symbols and punctuation.
* `sandhi_breaking.ipynb`: Jupyter Notebook for interactive testing and visualization.

## Key Features

* **Deep Learning Based:** Learns phonetic patterns directly from data rather than hard-coded rules.
* **Window-Based Isolation:** Higher accuracy by focusing the Seq2Seq model on the specific point of phonetic change.
* **Massive Corpus:** Trained on historical and grammatical datasets including Panini’s rules.
* **Cross-Platform Support:** Works with standard Devanagari input and outputs structured splits.

## Installation & Requirements

Ensure you have Python 3.12+ installed along with the following libraries:

```bash
pip install tensorflow tf_keras numpy scikit-learn indic-transliteration h5py
