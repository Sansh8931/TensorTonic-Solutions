import numpy as np
from collections import Counter

def tfidf_vectorizer(documents):
    """
    Convert text documents into TF-IDF feature matrix.
    
    Args:
        documents (list[str]): List of text documents.
    
    Returns:
        tuple: (tfidf_matrix, vocabulary)
            tfidf_matrix: np.ndarray of shape (n_docs, n_vocab)
            vocabulary: list[str] of unique terms, sorted alphabetically
    """
    # Handle empty corpus
    if not documents:
        return np.zeros((0, 0)), []

    # Tokenize documents (lowercase + split on whitespace)
    tokenized_docs = [doc.lower().split() for doc in documents]

    # Build vocabulary (sorted unique terms)
    vocab = sorted(set(term for doc in tokenized_docs for term in doc))
    vocab_index = {term: i for i, term in enumerate(vocab)}

    n_docs = len(documents)
    n_vocab = len(vocab)

    # Term frequency (TF)
    tf = np.zeros((n_docs, n_vocab), dtype=float)
    for d_idx, doc in enumerate(tokenized_docs):
        counts = Counter(doc)
        total_terms = len(doc)
        if total_terms == 0:
            continue
        for term, count in counts.items():
            tf[d_idx, vocab_index[term]] = count / total_terms

    # Document frequency (DF)
    df = np.zeros(n_vocab, dtype=int)
    for term, idx in vocab_index.items():
        df[idx] = sum(1 for doc in tokenized_docs if term in doc)

    # Inverse document frequency (IDF)
    # idf(t) = log(N / df(t)), with safe handling for df=0
    idf = np.zeros(n_vocab, dtype=float)
    for idx, freq in enumerate(df):
        if freq > 0:
            idf[idx] = np.log(n_docs / freq)
        else:
            idf[idx] = 0.0

    # TF-IDF = TF × IDF
    tfidf = tf * idf

    return tfidf, vocab
