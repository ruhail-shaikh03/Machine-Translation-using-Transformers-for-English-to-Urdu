# English-to-Urdu Machine Translation using Transformers

This project implements a Transformer-based neural machine translation (NMT) model to translate text from English to Urdu. The implementation follows the architecture proposed in the "Attention is All You Need" paper by Vaswani et al.

## Table of Contents

1.  [Objectives](#objectives)
2.  [Dataset](#dataset)
3.  [Project Structure (Example)](#project-structure-example)
4.  [Setup](#setup)
5.  [Implementation Details](#implementation-details)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Subword Tokenization](#subword-tokenization)
    *   [Transformer Model Architecture](#transformer-model-architecture)
6.  [Training](#training)
7.  [Evaluation](#evaluation)
8.  [Results (Observed)](#results-observed)
9.  [Running the Notebook](#running-the-notebook)
10. [Additional Tasks & Future Work](#additional-tasks--future-work)
11. [Acknowledgements](#acknowledgements)

## Objectives

*   Implement a Transformer-based model for English-to-Urdu machine translation.
*   Train the model on a suitable parallel corpus.
*   Evaluate the model's performance using appropriate metrics, primarily BLEU score.
*   Provide both quantitative results and qualitative analysis with example translations.

## Dataset

This project utilizes a parallel corpus for English-Urdu translation. The primary dataset used is:

*   **Parallel Corpus for English-Urdu Language:**
    *   Available at: [https://www.kaggle.com/datasets/zainuddin123/parallel-corpus-for-english-urdu-language](https://www.kaggle.com/datasets/zainuddin123/parallel-corpus-for-english-urdu-language) (specifically, the version linked in the notebook seems to be `/kaggle/input/parallel-corpus-for-english-urdu-language/Dataset/english-corpus.txt` and `urdu-corpus.txt`).
    *   This dataset contains over 24,000 sentence pairs. After cleaning and deduplication, the notebook uses approximately 24,000 unique sentence pairs.

Other potential resources (not explicitly used in the provided notebook but mentioned in objectives):
*   **UMC005: English-Urdu Parallel Corpus:** [https://ufal.mff.cuni.cz/umc/005-en-ur/](https://ufal.mff.cuni.cz/umc/005-en-ur/)

The dataset is split into training (85%) and validation (15%) sets.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/[your-username]/[your-repo-name].git
    cd eng-ur-transformer-nmt
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The notebook `gen ai q3.ipynb` uses TensorFlow and other libraries.
    ```bash
    pip install -r requirements.txt
    ```
    Create a `requirements.txt` file with the following (or based on your exact imports):
    ```
    tensorflow
    numpy
    pandas
    sentencepiece
    nltk  # For BLEU score calculation
    scikit-learn # For train_test_split
    matplotlib # For plotting
    # Add other specific libraries if used
    ```

4.  **Dataset:**
    *   Download the "Parallel Corpus for English-Urdu Language" from Kaggle or ensure the paths in the notebook (`ENG_CORPUS_PATH`, `URDU_CORPUS_PATH`) point to the correct locations.
    *   The notebook saves tokenizer models and the trained model to `/kaggle/working/`.

## Implementation Details

### Data Preprocessing
1.  **Loading:** English and Urdu corpora are loaded from text files.
2.  **Cleaning:**
    *   Removal of NaN values and empty strings.
    *   Deduplication of sentence pairs.
    *   **English Text:** Converted to lowercase, punctuation removed (simple regex used), extra whitespace removed.
    *   **Urdu Text:** English characters/numbers removed, specific English/common punctuation removed (while attempting to preserve Urdu punctuation like `۔`, `؟`, `،`), extra whitespace removed.
3.  **Splitting:** The cleaned dataset is split into training and validation sets.

### Subword Tokenization
*   **Method:** SentencePiece (Byte Pair Encoding - BPE type) is used for subword tokenization.
*   **Process:** Separate tokenizers are trained for English and Urdu using their respective cleaned training corpora.
*   **Vocabulary Size:** Target `VOCAB_SIZE` is set (e.g., 7000 for each language).
*   **Special Tokens:** `PAD_ID=0`, `BOS_ID=1` (Start Of Sentence), `EOS_ID=2` (End Of Sentence), `UNK_ID=3`.
*   **Encoding:**
    *   English sentences are tokenized.
    *   Urdu sentences are tokenized and `BOS` and `EOS` tokens are added.
*   **Padding:** Sequences are padded to `MAX_SEQUENCE_LENGTH` (e.g., 60) after batching.

### Transformer Model Architecture
The model follows the standard Transformer architecture ("Attention is All You Need"):

*   **Hyperparameters:**
    *   `D_MODEL`: 512 (Embedding and model dimension)
    *   `NUM_LAYERS`: 6 (Number of encoder/decoder layers)
    *   `NUM_HEADS`: 8 (Number of attention heads)
    *   `DFF`: 2048 (Dimension of the feed-forward network)
    *   `DROPOUT_RATE`: 0.1
*   **Components:**
    *   **Input Embeddings:** For both English (encoder) and Urdu (decoder).
    *   **Positional Encoding:** Added to embeddings to provide sequence order information.
    *   **Multi-Head Self-Attention:** Used in both encoder and decoder layers.
    *   **Scaled Dot-Product Attention:** The core attention mechanism.
    *   **Encoder:**
        *   Stack of `NUM_LAYERS` identical encoder layers.
        *   Each encoder layer has:
            *   Multi-Head Self-Attention sub-layer.
            *   Position-wise Feed-Forward Network sub-layer.
            *   Layer Normalization and Residual Connections around each sub-layer.
    *   **Decoder:**
        *   Stack of `NUM_LAYERS` identical decoder layers.
        *   Each decoder layer has:
            *   Masked Multi-Head Self-Attention sub-layer (to prevent attending to future tokens).
            *   Multi-Head Encoder-Decoder Attention sub-layer (attends to encoder output).
            *   Position-wise Feed-Forward Network sub-layer.
            *   Layer Normalization and Residual Connections around each sub-layer.
    *   **Final Linear Layer:** Projects decoder output to target vocabulary size (Urdu), followed by Softmax (implicitly handled by loss function).
*   **Masking:**
    *   **Padding Mask:** Used in both encoder and decoder to ignore padding tokens.
    *   **Look-Ahead Mask:** Used in the decoder's self-attention to ensure auto-regressive property.

## Training

*   **Optimizer:** AdamW optimizer with a custom learning rate schedule (`CustomSchedule` based on the Transformer paper) with `WARMUP_STEPS` (e.g., 1500). Weight decay and global clipnorm are also used.
*   **Loss Function:** Custom sparse categorical crossentropy loss function with **Label Smoothing** (e.g., 0.1) and masking for padding tokens.
*   **Metrics:** Custom `masked_accuracy` that ignores padding tokens.
*   **Data Preparation for `model.fit`:**
    *   Encoder input: English token IDs.
    *   Decoder input: Urdu token IDs, shifted right (starts with `BOS`, ends before `EOS`).
    *   Decoder output (target): Urdu token IDs, shifted left (starts after `BOS`, ends with `EOS`).
*   **Epochs:** Trained for `EPOCHS` (e.g., 30), but with early stopping.
*   **Batch Size:** `BATCH_SIZE` (e.g., 64).
*   **Callbacks:**
    *   `EarlyStopping`: Monitors `val_loss`, restores best weights.
    *   `ModelCheckpoint`: Saves the best model (entire model, not just weights) based on `val_loss` to `MODEL_SAVE_PATH`.
    *   `TensorBoard`: For logging training progress.

## Evaluation

*   **Primary Metric:** Corpus BLEU score, calculated on the validation set using `nltk.translate.bleu_score`.
    *   References and hypotheses are tokenized using the trained SentencePiece models for consistency.
    *   Smoothing function (e.g., `method4`) is applied.
*   **Qualitative Analysis:** Example translations are generated for a set of test English sentences using the trained model.
*   **Training Curves:** Plots of training and validation loss, and training and validation masked accuracy over epochs.
*   **Translation Function:** A `translate_sentence` (greedy decoding) and `translate_batch` function are implemented for generating translations.

## Results (Observed)

*(This section should be filled with the actual results from your notebook runs. The provided notebook shows some sample outputs, but the final BLEU score and curves would be key.)*

*   **Tokenization:**
    *   English Vocabulary Size: [Actual size from `en_tokenizer.get_piece_size()`]
    *   Urdu Vocabulary Size: [Actual size from `ur_tokenizer.get_piece_size()`]
*   **Training Performance:**
    *   Best Validation Loss: [e.g., ~3.54 from notebook output] achieved at Epoch [X].
    *   Best Validation Masked Accuracy: [e.g., ~53.75% from notebook output] achieved at Epoch [Y].
    *   Early stopping triggered at Epoch [Z].
*   **Quantitative Evaluation:**
    *   **Validation Corpus BLEU Score:** [e.g., ~3.55 from notebook output]. This score indicates the quality of translation; higher is better (typically on a 0-100 scale if multiplied by 100). A score of 3.55 is relatively low, suggesting room for improvement (common for NMT without extensive data/tuning).
*   **Qualitative Examples:**
    *   *(Include a few good and bad translation examples from your notebook output to illustrate model performance.)*
    *   E.g.,
        *   English: "i love kids" -> Urdu (Actual): "میں بچوں سے محبت کرتا ہوں" -> Urdu (Predicted): "مجھے بچوں سے پیار ہیں۔" (Good)
        *   English: "where is father" -> Urdu (Actual): "باپ کہاں ہے" -> Urdu (Predicted): "کہاں کہاں کہاں..." (Repetitive/Poor)
*   **Training Curves:**
    *   *(Describe the trend of loss and accuracy curves. Did they converge? Was there overfitting?)*

## Running the Notebook

1.  Ensure all dependencies from `requirements.txt` are installed in your Python environment.
2.  **Dataset:**
    *   If running on Kaggle, ensure the "Parallel Corpus for English-Urdu Language" dataset is added to your notebook environment. The notebook uses paths like `/kaggle/input/parallel-corpus-for-english-urdu-language/...`.
    *   If running locally, download the dataset and update `ENG_CORPUS_PATH` and `URDU_CORPUS_PATH` in the notebook.
3.  **Working Directory:**
    *   The notebook saves tokenizers, temporary training files, logs, and the best model to `/kaggle/working/`. Ensure this directory is writable or adjust paths as needed for local execution.
4.  Open and run the `gen ai q3.ipynb` notebook in a Jupyter environment (Jupyter Lab, Jupyter Notebook, Google Colab, Kaggle Notebooks).
5.  The notebook is structured to:
    *   Configure hyperparameters.
    *   Load, clean, and split the data.
    *   Train SentencePiece tokenizers.
    *   Prepare TensorFlow datasets.
    *   Define the Transformer model components (Positional Encoding, MHA, Encoder, Decoder, Transformer).
    *   Set up the training loop (optimizer, loss, metrics, callbacks).
    *   Train the model.
    *   Evaluate the model (BLEU score, example translations, plotting training history).
6.  **GPU Usage:** Using a GPU-enabled environment is **highly recommended** for training Transformer models due to their computational demands. The notebook is configured to use a GPU if available (e.g., on Kaggle or Colab).
7.  **Model Loading Issue:** The notebook output indicates an error when loading the saved model due to `Lambda` layers with Python lambda functions. To properly load the model, either:
    *   Pass `safe_mode=False` to `tf.keras.models.load_model` (if you trust the source).
    *   Define the lambda functions as regular Python functions and register them in `custom_objects`.
    *   For the evaluation part, the notebook falls back to using the `transformer_model` instance in memory if loading fails.

## Additional Tasks & Future Work

*   **Pre-trained Models:** Fine-tune a pre-trained multilingual NMT model like mBART or NLLB on this dataset and compare its performance.
*   **Data Augmentation:** Implement techniques like back-translation to increase training data.
*   **Beam Search Decoding:** Implement beam search instead of greedy decoding in the `translate_sentence` function for potentially better translation quality.
*   **Hyperparameter Tuning:** More extensive tuning of `D_MODEL`, `NUM_LAYERS`, `NUM_HEADS`, `DFF`, `DROPOUT_RATE`, learning rate schedule, and batch size.
*   **Larger Dataset:** Train on a larger and more diverse English-Urdu parallel corpus if available.
*   **Advanced Tokenization:** Explore other subword tokenization strategies or larger vocabularies.
*   **Regularization:** Experiment with different dropout rates or other regularization techniques.

## Acknowledgements

*   "Attention is All You Need" by Vaswani et. al. (Original Transformer paper).
*   TensorFlow and Keras libraries.
*   SentencePiece library for tokenization.
*   NLTK library for BLEU score calculation.
*   Kaggle dataset "Parallel Corpus for English-Urdu Language" by Zainuddin.
