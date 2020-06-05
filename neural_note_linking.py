import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

def get_tokenizer_and_model(model_name):
    """Get huggingface tokenizer and model for specified model name.
    
    Args:
        model_name (str): Name of the model to import.
    Returns:
        transformers Tokenizer object
        transformers Model object
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    
    return tokenizer, model

def get_text_embedding(text, tokenizer, model, aggregation="mean"):
    """Get a lanaguage-model-generated embedding for a bit of text.
    Texts longer than the model's size limit will be separated into several sequences and the
    embedding vectors averaged. Effect uncertain.
    
    Certain models might not work with this function, but all BERT-based models do.
    The recommended model for multilingual embedding is "xlm-roberta-base".
    
    Args:
        text (str): Text to get embedding for.
        tokenizer (transformers.Tokenizer): Tokenizer returned by get_tokenizer_and_model.
        model (transformers.Model): Model returned by get_tokenizer_and_model.
        aggregation (str): How to aggregate a sequence's embeddings.
            If "mean", take the mean of the embeddings of the tokens.
            If "first", take the embedding of the first token.
            Default: "mean".
    
    Returns:
        np.array: Embedding vector for the text. Dimension depends on model but is length-independent.
    """
    # Define aggregation function
    aggregation_functions = {
        "mean": lambda x: x.mean(axis=1).squeeze(),
        "first": lambda x: x[:, 0].squeeze()
    }
    assert aggregation in aggregation_functions, f"Unrecognized aggregation type: {aggregation}."
    aggregation_function = aggregation_functions[aggregation]
    
    # Convert text to tokens
    tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    
    # Run test to determine how many special tokens are added to a sequence
    test_token = tokens[0]
    num_special_tokens = len(tokenizer.prepare_for_model([test_token])["input_ids"]) - 1
    
    # Separate into sequences in case length is too large
    sequences = [
        tokens[i:i+tokenizer.max_len - num_special_tokens]
        for i in range(0, len(tokens), tokenizer.max_len)
    ]
    
    # Add special tokens to each sequence
    sequences = [
        torch.tensor([
            tokenizer.prepare_for_model(
                seq
            )["input_ids"]
        ])
        for seq in sequences
    ]
    
    # Compute embeddings (first output of the model = last hidden state, of shape (1, seq_length, hidden_size))
    embeddings = np.array([
        model(seq)[0].detach().numpy()
        for seq in sequences
    ])
    
    length_independent_embeddings = [
        aggregation_function(emb) # Should be of shape (hidden_size,)
        for emb in embeddings
    ]
    
    embedding = np.array(length_independent_embeddings).mean(axis=0)
    
    return embedding

def get_distance_matrix(vectors, metric="cosine", triangular=True, verbose=False):
    """Compute the distance between all pairs of vectors in a list and return them as a matrix.
    Nothing fancy here, so number of operations is quadratic in len(vectors).
    
    Args:
        vectors (list of 1-d arrays): Vectors to compare. All vectors should be of the same length.
        metric (str): Distance metric to use.
            Currently supported: "l2", "cosine". (cosine distance = 1 - cosine similarity, normalized to be between 0 and 1)
            Default: "cosine".
        triangular (bool): Whether to only fill the lower triangle of the matrix instead of returning a symmetric matrix.
            Default: True.
        verbose (bool): Whether to print progressbar.
            Default: False.
    
    Returns:
        np.array: Distance matrix of shape (len(vectors), len(vectors)).
    """
    distance_functions = {
        "l2": lambda x, y: np.linalg.norm(x-y),
        "cosine": lambda x, y: 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)) + 1) / 2
        # "cosine": lambda x, y: -np.log((np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)) + 1) / 2)
    }
    
    assert metric in distance_functions, f"Unrecognized metric: {metric}."
    
    distance_function = distance_functions[metric]
    
    n = len(vectors)
    pairs = list(itertools.combinations(range(n), 2))
    matrix = np.zeros((n, n))
    
    # The distance functions have as minimal distance 0, so we never compute the distance between a vector and itself.
    for i, j in tqdm(pairs, disable=not verbose):
        dist = distance_function(vectors[i], vectors[j])
        matrix[j, i] = dist
        if not triangular:
            matrix[i, j] = dist
    
    return matrix

def get_text_distances(texts, names=None, tokenizer=None, model=None, embedding_aggregation="mean", metric="cosine", skip_duplicates=True):
    """Get distances between texts as a pd.Series.
    
    Args:
        texts (list of strings): Texts to compare.
        names (list of strings): Names of texts. If None, use integers.
            Default: None.
        tokenizer (transformers.Tokenizer): Tokenizer returned by get_tokenizer_and_model. If None, will automatically load "xlm-roberta-base".
            Default: None.
        model (transformers.Model): Model returned by get_tokenizer_and_model. If None, will automatically load "xlm-roberta-base".
            Default: None.
        embedding_aggregation (str): How to aggregate each sequence's embeddings.
            If "mean", take the mean of the embeddings of the tokens.
            If "first", take the embedding of the first token.
            Default: "mean".
        metric (str): Distance metric to use.
            Currently supported: "l2", "cosine". (cosine distance = 1 - cosine similarity, normalized to be between 0 and 1)
            Default: "cosine".
        skip_duplicates (bool): Whether to skip rows representing a distance already present in the Series.
            If False, each distance will be represented twice, as srs.loc[a, b] and srs.loc[b, a].
            Default: True.
    
    Returns:
        pd.Series: Series with MultiIndex representing pairs of texts, and distances as values.
    """
    if (tokenizer is None) or (model is None):
        tokenizer, model = get_tokenizer_and_model("xlm-roberta-base")
    
    if names is None:
        names = range(len(texts))
    
    embeddings = [get_text_embedding(text, tokenizer, model, aggregation=embedding_aggregation) for text in tqdm(texts)]
    
    dist = pd.DataFrame(
        get_distance_matrix(embeddings, metric=metric, triangular=skip_duplicates),
        index=names,
        columns=names
    ).unstack()
    
    if skip_duplicates:
        # Keep only one row for each pair, also excluding the diagonal
        dist = dist.loc[itertools.combinations(names, 2)]
    else:
        # Exclude only the diagonal
        dist = dist.drop([(name, name) for name in names])
    
    return dist
    
