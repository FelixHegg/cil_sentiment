import re
import nltk
import numpy as np
import gensim
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer # Added for HF tokenization

import config

nltk_resources = ['punkt', 'punkt_tab']

def download_nltk_resources():
    """Downloads NLTK 'punkt' and 'punkt_tab' resources if not found."""
    print("Checking/downloading NLTK resources...")
    for resource in nltk_resources:
        try:
            if resource == 'punkt':
                 _ = nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading NLTK resource: {resource}...")
            try:
                nltk.download(resource, quiet=False)
                print(f"'{resource}' downloaded successfully.")
            except Exception as e:
                print(f"Error downloading '{resource}': {e}")

download_nltk_resources()

def simple_tokenizer(text):
    """Lowercase, remove non-alphanumerics, and tokenize text using NLTK."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[\d\W]+', ' ', text.lower()).strip()
    tokens = word_tokenize(text)
    return tokens

def load_embedding_model(name='word2vec-google-news-300'):
    """Loads a pre-trained gensim word embedding model (e.g., Word2Vec, FastText)."""
    print(f"Loading pre-trained embedding model ({name})...")
    try:
        model = api.load(name)
        if not hasattr(model, 'wv'): model.wv = model
        if not hasattr(model, 'vector_size'):
             if hasattr(model, 'wv') and hasattr(model.wv, 'vector_size'): model.vector_size = model.wv.vector_size
             else:
                 if '300' in name: model.vector_size = 300
                 elif '100' in name: model.vector_size = 100
                 else: raise AttributeError("Could not determine vector_size for gensim model")
        print(f"Gensim model '{name}' loaded. Vector size: {model.vector_size}")
        return model
    except ValueError as e:
         print(f"Error: Gensim model '{name}' not found or download failed. Gensim error: {e}")
         raise
    except Exception as e:
        print(f"An unexpected error occurred loading gensim model '{name}': {e}")
        raise

def load_hf_tokenizer(hf_model_name):
    """Loads a Hugging Face AutoTokenizer from a pre-trained model name."""
    print(f"Loading Hugging Face tokenizer for '{hf_model_name}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        print(f"Hugging Face tokenizer '{hf_model_name}' loaded.")
        return tokenizer
    except Exception as e:
        print(f"Error loading Hugging Face tokenizer '{hf_model_name}': {e}")
        raise

def vectorize_sentence_average(tokens, model):
    """Averages word vectors of tokens in a sentence; returns zero vector for empty/OOV sentences."""
    vectors = []
    for word in tokens:
        try:
             if hasattr(model, 'wv'): vectors.append(model.wv[word])
             else: vectors.append(model[word])
        except KeyError: 
            continue
    if not vectors: 
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

def calculate_max_seq_len(tokenized_sentences, percentile=config.SEQUENCE_LENGTH_PERCENTILE):
    """Calculates max sequence length based on a given percentile of sentence lengths."""
    print(f"Calculating sequence length based on {percentile}th percentile...")
    sentence_lengths = [len(tokens) for tokens in tokenized_sentences]
    if not sentence_lengths: return config.MIN_SEQUENCE_LENGTH
    max_len = int(np.percentile(sentence_lengths, percentile))
    max_len = max(max_len, config.MIN_SEQUENCE_LENGTH)
    print(f"Calculated MAX_SEQ_LEN for non-HF: {max_len}")
    return max_len

def vectorize_sentence_sequence(tokens, model, max_len):
    """Converts tokens to a sequence of word vectors, with padding/truncation."""
    vectors = []
    for word in tokens:
        try:
            if word in model: 
                vectors.append(model[word])
        except KeyError: 
            continue
    vectors = vectors[:max_len]
    num_vectors = len(vectors)
    if num_vectors < max_len:
        padding_size = max_len - num_vectors
        if not hasattr(model, 'vector_size'): 
            raise AttributeError("Model lacks vector_size attribute")
        padding = np.zeros((padding_size, model.vector_size), dtype=np.float32)
        vectors_np = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, model.vector_size), dtype=np.float32)
        vectors = np.concatenate((vectors_np, padding), axis=0)
    elif num_vectors == 0: 
        return np.zeros((max_len, model.vector_size), dtype=np.float32)
    else: 
        vectors = np.array(vectors, dtype=np.float32)
    if vectors.shape != (max_len, model.vector_size):
        print(f"Warning: Vector shape mismatch. Expected {(max_len, model.vector_size)}, got {vectors.shape}. Returning zeros.")
        return np.zeros((max_len, model.vector_size), dtype=np.float32)
    return vectors

def get_label_utils(labels):
    """Fits LabelEncoder and creates a mapping for MAE-compatible labels."""
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    num_classes = len(label_encoder.classes_)
    print(f"Label mapping (class -> index): {list(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    try:
        index_to_mae_label_map = {
            label_encoder.transform(['negative'])[0]: -1,
            label_encoder.transform(['neutral'])[0]: 0,
            label_encoder.transform(['positive'])[0]: 1
        }
        print(f"Index to MAE Label Map: {index_to_mae_label_map}")
    except ValueError as e:
        print(f"Warning: Could not create MAE map. Ensure labels include 'negative', 'neutral', 'positive'. Error: {e}")
        index_to_mae_label_map = {}
    return label_encoder, index_to_mae_label_map, num_classes

def preprocess_data_non_hf(sentences, vectorization_strategy, embedding_model_name, max_seq_len_non_hf=None):
    """Tokenizes and vectorizes sentences for non-Hugging Face models (Word2Vec/FastText)."""
    embedding_model = load_embedding_model(embedding_model_name) # Load gensim model
    print("Tokenizing sentences with simple_tokenizer for non-HF...")
    tokenized_sentences = [simple_tokenizer(sent) for sent in sentences]

    if vectorization_strategy == 'average':
        print("Vectorizing sentences using average strategy...")
        vectors = np.array([vectorize_sentence_average(tokens, embedding_model)
                           for tokens in tokenized_sentences])
        print(f"Average vectors created with shape: {vectors.shape}")
        return vectors, None # No max_seq_len used for average

    elif vectorization_strategy == 'sequence':
        if max_seq_len_non_hf is None:
             max_seq_len_non_hf = calculate_max_seq_len(tokenized_sentences)
        else:
             print(f"Using provided MAX_SEQ_LEN for non-HF: {max_seq_len_non_hf}")

        print(f"Vectorizing sentences as sequences (padding/truncating to {max_seq_len_non_hf})...")
        sequences = np.array([vectorize_sentence_sequence(tokens, embedding_model, max_seq_len_non_hf)
                              for tokens in tokenized_sentences])
        print(f"Sequence vectors created with shape: {sequences.shape}")
        return sequences, max_seq_len_non_hf
    else:
        raise ValueError(f"Unknown vectorization strategy for non-HF: {vectorization_strategy}")

def preprocess_data_hf(sentences, hf_tokenizer_name, max_seq_len):
    """Tokenizes sentences using a specified Hugging Face tokenizer for Transformer models."""
    tokenizer = load_hf_tokenizer(hf_tokenizer_name)
    print(f"Tokenizing all sentences using HF tokenizer '{hf_tokenizer_name}' (padding/truncating to {max_seq_len})...")
    encodings = tokenizer(
        sentences,
        truncation=True,
        padding='max_length',
        max_length=max_seq_len,
        return_tensors='pt' # Return PyTorch tensors
    )
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    print(f"HF Input IDs shape: {input_ids.shape}, Attention Masks shape: {attention_masks.shape}")
    return input_ids, attention_masks, max_seq_len # Return max_seq_len used
