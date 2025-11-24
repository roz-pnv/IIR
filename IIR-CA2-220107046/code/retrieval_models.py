from collections import defaultdict, Counter
import math
from tqdm import tqdm


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


def jelinek_mercer_score(query, inverted_index, lambda_param=0.7):
    """
    Jelinek-Mercer (Linear Interpolation) smoothing.

    log P(q|d) = sum_w log( (1-λ)*P_ML(w|d) + λ*P(w|C) )
               = sum_w log(λ*pC) + log(1 + ((1-λ)/λ) * tf/(|d|*pC))
    The constant log(λ*pC) is ignored since it's query-independent.
    """
    #YOUR CODE HERE
    scores = defaultdict(float)
    
    for term in query.tokens:
        P_w_C = inverted_index.compute_collection_prob(term)
        postings = inverted_index.get_postings(term)
        
        for doc_id, c_w_d in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            P_w_d = c_w_d / doc_len
            
            smoothed_prob = (1 - lambda_param) * P_w_d + lambda_param * P_w_C
       
            scores[doc_id] += _safe_log(smoothed_prob)
    
    return scores


def dirichlet_prior_score(query, inverted_index, mu=2000):
    """
    Dirichlet prior smoothing.

    p(w|d) = (c(w,d) + μ*pC) / (|d| + μ)
    log P(q|d) = sum_w log(1 + tf/(μ*pC)) + |q| * log(μ / (|d| + μ))
    """
    #YOUR CODE HERE
    scores = defaultdict(float)
    
    for term in query.tokens:
        P_w_C = inverted_index.compute_collection_prob(term)
        postings = inverted_index.get_postings(term)
        
        for doc_id, c_w_d in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            
            smoothed_prob = (c_w_d + mu * P_w_C) / (doc_len + mu)
            
            scores[doc_id] += _safe_log(smoothed_prob)
    
    return scores


def absolute_discounting_score(query, inverted_index, delta=0.7):
    """
    Absolute discounting smoothing.

    p(w|d) = max(tf - δ, 0)/|d| + δ*|d|_u/|d| * pC
    log p(w|d) ≈ log(δ*|d|_u/|d|*pC) + log(1 + max(tf-δ,0)/(δ*|d|_u*pC))
    """
    #YOUR CODE HERE
    scores = defaultdict(float)
    
    for term in query.tokens:
        P_w_C = inverted_index.compute_collection_prob(term)
        postings = inverted_index.get_postings(term)
        
        for doc_id, c_w_d in postings:
            doc_len = inverted_index.doc_lengths[doc_id]

            c_w_d_discounted = max(c_w_d - delta, 0)

            doc_u_count = inverted_index.term_doc_freq[term]
            
            smoothed_prob = (c_w_d_discounted / doc_len) + (delta * doc_u_count / doc_len) * P_w_C
            
            scores[doc_id] += _safe_log(smoothed_prob)
    
    return scores


def _safe_log(x, eps=1e-12):
    return math.log(x if x > eps else eps)

def _safe_prob(x, eps=1e-12):
    return x if x > eps else eps

def compute_doc_language_model(inverted_index, doc_id, mu=2000):
    """
    Dirichlet-smoothed document language model (sparse over doc terms):
        p(w|θ_d) = (tf + μ * p(w|C)) / (|d| + μ)
    """
    tokens = inverted_index.doc_tokens[doc_id]
    dlen = len(tokens)
    term_counts = Counter(tokens)
    doc_model = {}
    for w, tf in term_counts.items():
        pc = _safe_prob(inverted_index.compute_collection_prob(w))
        doc_model[w] = (tf + mu * pc) / (dlen + mu)
    return doc_model


def mixture_model_prf(query, inverted_index, top_n=10, num_expansion_terms=10,
                      max_iterations=50, convergence_threshold=1e-5,
                      lambda_init=0.5):
    """
    Pseudo Relevance Feedback using the Mixture Model with EM.
    Expands the input query in-place via query.expand(expansion_terms).
    Steps:
      1) Initial retrieval (BM25) → top_n docs
      2) Aggregate term counts over feedback docs
      3) Build background P_bg
      4) Initialize topic model P_w_R and λ
      5) EM to refine P_w_R and λ
      6) Pick top-K expansion terms (not in original query), normalize, expand
    """
    # 1) initial retrieval
    initial_scores = bm25_score(query, inverted_index)  # <- ensure bm25_score is imported into global scope
    ranked_docs = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [doc_id for doc_id, _ in ranked_docs[:top_n]]
    if not top_docs:
        return query

    # 2) feedback stats
    term_counts = Counter()
    total_top_terms = 0
    for doc_id in top_docs:
        tokens = inverted_index.doc_tokens[doc_id]
        term_counts.update(tokens)
        total_top_terms += len(tokens)
    if total_top_terms == 0:
        return query

    # 3) background model
    vocab = set(term_counts.keys())
    min_prob = 1e-12
    P_bg = {t: max(inverted_index.compute_collection_prob(t), min_prob) for t in vocab}

    # 4) init topic model & lambda
    P_w_R = {t: term_counts[t] / total_top_terms for t in vocab}
    lambda_val = float(lambda_init)

    # 5) EM
    for _ in tqdm(range(max_iterations), desc="EM Iterations", unit="iter"):
        P_w_R_old = P_w_R.copy()

        # --------- E-step (gamma[t] = p(z=1|t) background responsibility) ---------
        #YOUR CODE HERE
        P_z_1_given_w = {}
        for word in vocab:
            P_z_1_given_w[word] = (lambda_val * P_bg[word]) / (lambda_val * P_bg[word] + (1 - lambda_val) * P_w_R[word])

        # --------- M-step (update P_w_R and λ) ---------
        #YOUR CODE HERE
        term_count_sum = 0
        for term in vocab:
            term_count_in_docs = sum([inverted_index.get_postings(doc_id).count(term) for doc_id in top_docs])
            term_count_sum += term_count_in_docs * (1 - P_z_1_given_w[term])

        for term in vocab:
            P_w_R[term] = (term_counts[term] * (1 - P_z_1_given_w[term])) / term_count_sum

        # convergence check
        diff = sum(abs(P_w_R[t] - P_w_R_old[t]) for t in vocab)
        if diff < convergence_threshold:
            break

    # 6) select expansion terms (not in original)
    original = set(query.tokens)
    candidates = sorted(P_w_R.items(), key=lambda x: x[1], reverse=True)
    expansion_terms = {}
    for t, p in candidates:
        if t not in original:
            expansion_terms[t] = p
        if len(expansion_terms) >= num_expansion_terms:
            break

    # normalize weights to [0,1]
    if expansion_terms:
        m = max(expansion_terms.values())
        if m > 0:
            expansion_terms = {t: p/m for t, p in expansion_terms.items()}

    query.expand(expansion_terms)
    return query


def divergence_minimization_prf(query, inverted_index, top_n=10,
                                num_expansion_terms=15, lambda_param=0.1, mu=2000):
    """
    Divergence Minimization Pseudo-Relevance Feedback (DM-PRF)
    -----------------------------------------------------------
    Based on Zhai & Lafferty (2001):
        p(w|θ̂_F) ∝ exp( [1/((1-λ)|F|)] * Σ_i log p(w|θ_i)
                        - [λ/(1-λ)] * log p(w|C) )
    """

    # 1) Initial retrieval
    initial_scores = bm25_score(query, inverted_index)
    ranked_docs = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [doc_id for doc_id, _ in ranked_docs[:top_n]]
    if not top_docs:
        return query

    # 2) Build document language models
    doc_models = {d: compute_doc_language_model(inverted_index, d, mu=mu) 
                  for d in top_docs}

    # 3) Collect vocabulary from feedback docs
    vocab = set()
    for d in top_docs:
        vocab.update(doc_models[d].keys())

    eps = 1e-12
    log_terms = {}
    F_size = len(top_docs)  # |F|

    
    #YOUR CODE HERE
    # TODO: For each word in vocabulary:
    #   1. Compute Σ_i log p(w|θ_i) across all feedback documents
    #   2. Get log p(w|C) from collection
    #   3. Apply DM formula: [1/((1-λ)|F|)] * Σ - [λ/(1-λ)] * log p(w|C)
    #   4. Store result in log_terms[w]
    # 
    # Hint: Use math.log() and be careful with zero probabilities (use eps)
    for term in vocab:
        sum_log_p_w_theta = 0
        for doc_id in top_docs:
            doc_model = doc_models[doc_id]
            p_w_given_doc = doc_model.get(term, eps)
            sum_log_p_w_theta += math.log(p_w_given_doc)

        p_w_collection = inverted_index.compute_collection_prob(term)
        p_w_collection = max(p_w_collection, eps)
        log_p_w_collection = math.log(p_w_collection)

        log_terms[term] = (1 / ((1 - lambda_param) * F_size)) * sum_log_p_w_theta - (lambda_param / (1 - lambda_param)) * log_p_w_collection


    # 5) Normalize using log-sum-exp
    max_log = max(log_terms.values())
    Z = max_log + math.log(sum(math.exp(v - max_log) for v in log_terms.values()))
    feedback_model = {w: math.exp(v - Z) for w, v in log_terms.items()}

    #YOUR CODE HERE
    # TODO: 
    #   1. For each word in feedback_model (except original query terms):
    #      - Compute KL divergence score: log p(w|θ̂_F) - log p(w|C)
    #      - Store as (word, score) tuple
    #   2. Sort terms by score (descending)
    #   3. Select top num_expansion_terms
    #
    # Hint: Skip words that are already in query.tokens
    chosen = chosen[:num_expansion_terms]
    chosen = []
    query_terms = set(query.tokens)

    for term, score in feedback_model.items():
        if term not in query_terms:
            kl_divergence_score = math.log(score) - log_p_w_collection
            chosen.append((term, kl_divergence_score))

    chosen = sorted(chosen, key=lambda x: x[1], reverse=True)
    chosen = chosen[:num_expansion_terms]


    # ========================================
    # END OF YOUR CODE
    # ========================================

    # 7) Normalize and expand query
    if chosen:
        max_score = max(s for _, s in chosen)
        expansion_terms = {w: (s / max_score if max_score > 0 else 0.0) 
                          for w, s in chosen}
        query.expand(expansion_terms)

    return query

def rm3_prf(query, inverted_index, top_n=10, num_expansion_terms=15, alpha=0.6):
    """
    RM3: Regularized Relevance Model
    
    Args:
        query: Query object
        inverted_index: InvertedIndex object
        top_n: Number of feedback documents
        num_expansion_terms: Number of terms to add
        alpha: Interpolation weight for feedback model (0.5-0.8)
    
    Returns:
        Expanded query with regularization
    """
    # YOUR CODE HERE
    initial_scores = bm25_score(query, inverted_index)
    ranked_docs = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)
    top_docs = [doc_id for doc_id, _ in ranked_docs[:top_n]]
    if not top_docs:
        return query

    doc_language_models = {
        doc_id: compute_doc_language_model(inverted_index, doc_id, mu=2000)
        for doc_id in top_docs
    }

    all_terms = set()
    for doc_id in top_docs:
        all_terms.update(doc_language_models[doc_id].keys())

    query_weights = {term: 1 if term in query.tokens else 0 for term in all_terms}

    feedback_weights = {}
    for term in all_terms:
        log_sum = 0
        for doc_id in top_docs:
            doc_model = doc_language_models[doc_id]
            log_sum += math.log(doc_model.get(term, 1e-12))
        feedback_weights[term] = math.exp(log_sum / len(top_docs))

    combined_weights = {
        term: alpha * query_weights.get(term, 0) + (1 - alpha) * feedback_weights.get(term, 0)
        for term in all_terms
    }

    expansion_terms = {
        term: score for term, score in sorted(combined_weights.items(), key=lambda x: x[1], reverse=True)
        if term not in query.tokens
    }

    expansion_terms = dict(list(expansion_terms.items())[:num_expansion_terms])

    if expansion_terms:
        max_score = max(expansion_terms.values())
        normalized_expansion_terms = {term: score / max_score for term, score in expansion_terms.items()}
        query.expand(normalized_expansion_terms)

    return query
