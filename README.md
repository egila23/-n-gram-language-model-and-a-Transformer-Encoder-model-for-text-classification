# NLP: Language Modeling

**Area**  Natural Language Processing (Spring 2026)  
**Released:** February 19, 2026  
**Due:** March 12, 2026

## Overview

This project focuses on building language models to understand how to generate coherent text sequences. You will implement:

1. **N-Gram Language Models** - Traditional statistical approaches using n-grams and smoothing
2. **Transformer Encoder Classifier** - Modern deep learning approach using transformer architectures for text classification

## Project Structure

```
├── HW2.ipynb                      # Main Jupyter notebook with exercises and test cases
├── encoder_classifier.py          # Transformer encoder-based text classifier implementation
├── ngram_lm.py                    # N-gram language model implementation
├── utils.py                       # Utility functions for data processing (read-only)
├── glove.6B.50d.txt              # Pre-trained GloVe embeddings (50-dimensional)
├── answers.txt                    # Submission file for written answers
└── data/
    ├── sample.txt                 # Small sample file for testing
    ├── bbc/                       # BBC news articles dataset
    │   ├── business.txt
    │   ├── entertainment.txt
    │   ├── politics.txt
    │   ├── sport.txt
    │   └── tech.txt
    ├── lyrics/                    # Song lyrics dataset
    │   ├── billie_eillish.txt
    │   ├── taylor_swift.txt
    │   ├── green_day.txt
    │   ├── ed_sheeran.txt
    │   ├── and others...
    ├── train/                     # Training data split
    └── test/                      # Test data split
```

## Key Files

- **HW2.ipynb**: Main submission notebook containing:
  - Step-by-step guidance for implementing n-gram models
  - Examples of data preprocessing with START, END, and UNK tokens
  - Test cases for validation
  - Analysis questions
  
- **encoder_classifier.py**: Includes:
  - `PositionalEncoding`: Adds positional information to embeddings
  - `TransformerEncoderClassifier`: Text classification using transformer encoder
  
- **ngram_lm.py**: Includes:
  - `get_ngrams()`: Extract n-grams from token sequences
  - `NGramLanguageModel`: Language model with smoothing support
  
- **utils.py**: Provides helper functions:
  - `read_file()`: Load text data
  - `preprocess()`: Add START/END tokens for n-gram window
  - `flatten()`: Convert nested lists to 1D sequences
  - Special symbols: `START`, `EOS` (end of sentence), `UNK` (unknown)

## Data

The project uses two datasets:

1. **BBC News Articles** (bbc/): News texts for topic classification
   - Categories: business, entertainment, politics, sport, tech
   
2. **Song Lyrics** (lyrics/): Lyrics from various artists
   - Artists: Billie Eilish, Taylor Swift, Green Day, Ed Sheeran, and others

Data is preprocessed with:
- One sentence per line
- All punctuation removed
- Tokens separated by whitespace

## Getting Started

### Prerequisites

```bash
pip install torch
pip install transformers
pip install numpy
pip install requests
pip install tqdm
```

### Running the Code

1. **Work in the Jupyter Notebook:**
   ```bash
   jupyter notebook HW2.ipynb
   ```

2. **Implement Language Models:**
   - Follow the TODO sections in the notebook
   - Copy completed implementations to `ngram_lm.py` and `encoder_classifier.py`

3. **Test Your Implementation:**
   - Use the provided test cases in the notebook
   - Validate against sample data in `data/sample.txt`

## Assignment Components

### Part 1: N-Gram Language Models (60 points)

- **Step 0**: Data preprocessing
  - Load and preprocess text data
  - Handle special tokens
  - Create n-gram windows
  
- **Step 1**: N-gram language model implementation
  - Extract n-grams from sequences
  - Implement smoothing (Laplace, etc.)
  - Calculate probabilities and perplexity

### Part 2: Transformer Encoder Classifier (40 points)

- Implement transformer encoder with positional encoding
- Fine-tune with GloVe embeddings
- Classify texts by category

## Submission

**Programming Components:**
- Submit on Gradescope:
  - `ngram_lm.py`
  - `encoder_classifier.py`

**Written Components:**
- Analysis questions (answers in `answers.txt`)
- Questions require running your model implementations

## Notes

- **Do not edit** `utils.py` - it contains essential utility functions
- Use GloVe embeddings for encoder-based approaches
- Special symbols (`START`, `EOS`, `UNK`) are pre-defined in utils.py
- The notebook contains helpful examples for each section
- Start by implementing and testing in the notebook before copying to separate files

## Additional Resources

For transformer architecture details, refer to:
- Positional encoding formulas in `encoder_classifier.py`
- PyTorch documentation for `nn.TransformerEncoder`
- The assignment notebook for conceptual explanations
# -n-gram-language-model-and-a-Transformer-Encoder-model-for-text-classification
