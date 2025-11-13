# retrieval_models.py
from collections import defaultdict
import math

def bm25_score(query, inverted_index, k=1.5, b=0.75):
    """
    Compute BM25 scores for a given query against all documents in the inverted index.

    Parameters:
    - query: Query object
    - inverted_index: InvertedIndex object
    - k: BM25 k parameter (default=1.5)
    - b: BM25 b parameter (default=0.75)

    Returns:
    - scores: Dictionary mapping doc_id to BM25 score
    """
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k + 1)
            denominator = freq + k * (1 - b + b * (doc_len / avg_doc_len))
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score  # Incorporate term weight
    return scores

#YOUR CODE HERE (implement the retrieval models)

def method1_score(query, inverted_index):
    doc_scores = defaultdict(float)
    for token in query.tokens:
        for doc_id, freq in inverted_index.get_postings(token):
            doc_scores[doc_id] += inverted_index.compute_idf(token)
    return doc_scores


def method2_score(query, inverted_index, k=1.5):
    doc_scores = defaultdict(float)
    for token in query.tokens:
        for doc_id, freq in inverted_index.get_postings(token):
            value = ((k + 1) * freq) / (freq + k)
            doc_scores[doc_id] += value
    return doc_scores


def method3_score(query, inverted_index, k=1.5):
    doc_scores = defaultdict(float)
    for token in query.tokens:
        idf_val = inverted_index.compute_idf(token)
        for doc_id, freq in inverted_index.get_postings(token):
            dl = inverted_index.doc_lengths[doc_id]
            avg_dl = inverted_index.avg_doc_len
            normalization = k * (dl / avg_dl)
            doc_scores[doc_id] += idf_val * ((freq * (k + 1)) / (freq + normalization))
    return doc_scores


def method4_score(query, inverted_index):
    doc_scores = defaultdict(float)
    for token in query.tokens:
        for doc_id, freq in inverted_index.get_postings(token):
            if freq > 0:
                doc_scores[doc_id] += 1.0
    return doc_scores


def method5_score(query, inverted_index, k=1.5, b=0.75):
    doc_scores = defaultdict(float)
    for token in query.tokens:
        idf_val = inverted_index.compute_idf(token)
        for doc_id, freq in inverted_index.get_postings(token):
            dl = inverted_index.doc_lengths[doc_id]
            avg_dl = inverted_index.avg_doc_len
            denom = freq + k * ((1 - b) + b * (dl / avg_dl))
            score = (idf_val ** 2) * (freq / denom)
            doc_scores[doc_id] += score
    return doc_scores


def method6_score(query, inverted_index, k=1.5, b=0.75, delta=1.0):
    doc_scores = defaultdict(float)
    for token in query.tokens:
        idf_val = inverted_index.compute_idf(token)
        for doc_id, freq in inverted_index.get_postings(token):
            dl = inverted_index.doc_lengths[doc_id]
            avg_dl = inverted_index.avg_doc_len
            adjust = (1 - b) + b * (dl / avg_dl)
            numerator = freq * (k + 1)
            denom = freq + k * adjust
            doc_scores[doc_id] += idf_val * ((numerator / denom) + delta)
    return doc_scores


# Method 7 — Proposed Custom Scoring Function (BM25L Variant)
def method7_score(query, inverted_index, k=1.5, b=0.75):
    doc_scores = defaultdict(float)
    for token in query.tokens:
        idf_val = inverted_index.compute_idf(token)
        for doc_id, freq in inverted_index.get_postings(token):
            dl = inverted_index.doc_lengths[doc_id]
            avg_dl = inverted_index.avg_doc_len

            denom = freq + k * ((1 - b) + b * (dl / avg_dl))
            base_score = (idf_val * freq * (k + 1)) / denom

            length_penalty = 1 / (1 + math.log(1 + dl / avg_dl))
            doc_scores[doc_id] += base_score * length_penalty

    return doc_scores


def pivoted_length_v1_score(query, inverted_index, b=0.75):
    doc_scores = defaultdict(float)
    for token in query.tokens:
        idf_val = inverted_index.compute_idf(token)
        for doc_id, freq in inverted_index.get_postings(token):
            dl = inverted_index.doc_lengths[doc_id]
            avg_dl = inverted_index.avg_doc_len
            denom = (1 - b) + b * (dl / avg_dl)
            score = idf_val * (math.log(1 + math.log(1 + freq)) / denom)
            doc_scores[doc_id] += score
    return doc_scores


def pivoted_length_v2_score(query, inverted_index, b=0.75):
    doc_scores = defaultdict(float)
    for token in query.tokens:
        idf_val = inverted_index.compute_idf(token)
        for doc_id, freq in inverted_index.get_postings(token):
            dl = inverted_index.doc_lengths[doc_id]
            avg_dl = inverted_index.avg_doc_len
            denom = (1 - b) + b * (dl / avg_dl)
            score = idf_val * (math.log(1 + freq) / denom)
            doc_scores[doc_id] += score
    return doc_scores
