import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

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

def get_text_distances(texts, names=None, sentence_transformer=None, metric="cosine", skip_duplicates=True):
    """Get distances between texts as a pd.Series.
    
    Args:
        texts (list of strings): Texts to compare.
        names (list of strings): Names of texts. If None, use integers.
            Default: None.
        sentence_transformer (SentenceTransformer): Transformer for computing sentence embeddings.
            If None, will automatically load "distiluse-base-multilingual-cased"
        metric (str): Distance metric to use.
            Currently supported: "l2", "cosine". (cosine distance = 1 - cosine similarity, normalized to be between 0 and 1)
            Default: "cosine".
        skip_duplicates (bool): Whether to skip rows representing a distance already present in the Series.
            If False, each distance will be represented twice, as srs.loc[a, b] and srs.loc[b, a].
            Default: True.
    
    Returns:
        pd.Series: Series with MultiIndex representing pairs of texts, and distances as values.
    """
    if sentence_transformer is None:
        sentence_transformer = SentenceTransformer("distiluse-base-multilingual-cased")
    
    if names is None:
        names = range(len(texts))
    
    embeddings = sentence_transformer.encode(texts)
    
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
    
def get_small_dimensional_embeddings(texts, sentence_transformer=None, dimension=2):
    """Embed texts in small dimension using PCA.
    
    Args:
        texts (list of strings): Texts to compare.
        sentence_transformer (SentenceTransformer): Transformer for computing sentence embeddings.
            If None, will automatically load "distiluse-base-multilingual-cased"
            Default: None.
        dimension (int): How many dimensions to keep.
            Default: 2.
    
    Returns:
        np.array with shape (n_texts, dimension)
    """
    if sentence_transformer is None:
        sentence_transformer = SentenceTransformer("distiluse-base-multilingual-cased")
    
    embeddings = sentence_transformer.encode(texts)
    
    small_dim_embeddings = PCA(dimension).fit_transform(embeddings)
    
    return small_dim_embeddings
    
def scatter_with_annotations(coords, annotations):
    """Create a scatterplot based on some coords and adding text annotations.
    
    Args:
        coords (array-like): Coordinates to scatter and add text to.
            Each element must be 2-dimensional.
        annotations (list of strings): Texts to add to the plot.
            Must be of same length as coords.
    
    Returns:
        plt.Axes: Axes object for the plot.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.4)
    for name, vect in zip(annotations, coords):
        plt.annotate(name, vect)
    
    return ax

def embed_and_plot(texts, names=None, sentence_transformer=None):
    """Embed texts in 2 dimensions and plot them using their names.
    
    Args:
        texts (list of strings): Texts to embed.
        names (list of strings): Names of texts. If None, use integers.
            Default: None.
        sentence_transformer (SentenceTransformer): Transformer for computing sentence embeddings.
            If None, will automatically load "distiluse-base-multilingual-cased"
    
    Returns:
        plt.Axes: Axes object for the plot
    """
    if sentence_transformer is None:
        sentence_transformer = SentenceTransformer("distiluse-base-multilingual-cased")
    
    if names is None:
        names = range(len(texts))
    
    embeddings = get_small_dimensional_embeddings(texts, sentence_transformer=sentence_transformer, dimension=2)
    
    return scatter_with_annotations(embeddings, names)
