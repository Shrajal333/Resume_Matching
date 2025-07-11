import nltk
import bm25s
import string
import Stemmer
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
stemmer = Stemmer.Stemmer("english")
STOPWORDS = set(stopwords.words('english'))

def unit_normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

def remove_stopwords_and_stem(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    stems = stemmer.stemWords(tokens)
    return ' '.join(stems)

def compute_node_scores(docs, multiqueries, embed_model):
    node_scores = []

    for doc in docs:
        doc_embedding = embed_model.get_text_embedding(doc.text_resource.text)
        similarities = [
            embed_model.similarity(doc_embedding, embed_model.get_text_embedding(mq))
            for mq in multiqueries
        ]

        node_scores.append(max(similarities))
    return np.array(node_scores)

def compute_node_scores(docs, multiqueries, embed_model):
    multiquery_embeddings = np.stack([
        embed_model.get_text_embedding(mq)
        for mq in multiqueries
    ])

    doc_texts = [doc.text_resource.text for doc in docs]
    doc_embeddings = np.stack([
        embed_model.get_text_embedding(doc_text)
        for doc_text in doc_texts
    ])

    doc_embeddings_norm = unit_normalize(doc_embeddings)
    multiquery_embeddings_norm = unit_normalize(multiquery_embeddings)

    similarities = np.dot(doc_embeddings_norm, multiquery_embeddings_norm.T)
    node_scores = similarities.max(axis=1)
    return node_scores

def compute_bm25_filtered_scores(docs, multiqueries):

    corpus = [remove_stopwords_and_stem(doc.text) for doc in docs]
    corpus_tokens = bm25s.tokenize(corpus, stopwords=None, stemmer=None)
    
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    bm25_multi_scores = []
    for q in multiqueries:
        query_processed = remove_stopwords_and_stem(q)
        query_tokens = bm25s.tokenize(query_processed, stopwords=None, stemmer=None)

        all_scores = retriever.get_scores(list(query_tokens[1]))
        bm25_multi_scores.append(all_scores)

    bm25_multi_scores = np.array(bm25_multi_scores)
    bm25_all = np.max(bm25_multi_scores, axis=0)

    return bm25_all

def jaccard_scores(query, candidates):
    query = remove_stopwords_and_stem(query)
    candidates = [remove_stopwords_and_stem(c) for c in candidates]

    vect = CountVectorizer(binary=True)
    all_texts = [query] + candidates

    X = vect.fit_transform(all_texts)
    qvec = X[0].toarray()[0]
    scores = []

    for i in range(1, X.shape[0]):
        dvec = X[i].toarray()[0]

        if np.sum(qvec) == 0 and np.sum(dvec) == 0:
            scores.append(1.0)
        else:
            scores.append(jaccard_score(qvec, dvec))
    return np.array(scores)

def compute_jaccard_filtered_scores(multiqueries, candidate_texts):
    jaccard_multi_scores = []
    
    for q in multiqueries:
        scores = jaccard_scores(q, candidate_texts)
        jaccard_multi_scores.append(scores)

    jaccard_multi_scores = np.array(jaccard_multi_scores)
    jaccard_all = np.max(jaccard_multi_scores, axis=0)

    return jaccard_all