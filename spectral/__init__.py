"""Spectral graph analysis of attention-based compression.

See IMPLEMENTATION_PLAN.md for the full design. Phase 1 (this milestone)
delivers the plumbing: attention extraction and the token -> sentence graph
reduction.
"""

from spectral.graph_builder import (
    aggregate_attention,
    final_token_attention,
    sentence_membership,
    pool_sentence_graph,
    symmetrize,
    sparsify,
    query_attention_per_sentence,
    build_sentence_graph,
)
from spectral.laplacian import (
    degree_vector,
    eigh_laplacian,
    induced_subgraph,
    laplacian,
    normalized_laplacian,
    pseudoinverse,
)
from spectral.h1_test import (
    align_sentences_by_text,
    davis_kahan_angle,
    edge_rank_corr,
    effective_resistance_drift,
    effective_resistance_matrix,
    laplacian_spectrum,
    spectral_entropy,
    spectrum_distance,
    run_subgraph_hypothesis,
    top_edge_recall,
)
from spectral.ncut import (
    cheeger_ratio,
    cut_value,
    jaccard,
    ncut_q,
    query_mass,
    volume,
)
from spectral.cut_solvers import (
    anchored_spectral_cut,
    exact_min_ncut,
    fiedler_vector,
    local_search,
    query_anchored_spectral_cut,
    random_cut,
    spectral_cut,
    top_query_attention_cut,
)
from spectral.h2_test import (
    algebraic_connectivity,
    run_optimal_cut_hypothesis,
)
from spectral.streaming_extractor import (
    StreamedGraph,
    StreamingExtractor,
)

__all__ = [
    # graph construction
    "aggregate_attention",
    "final_token_attention",
    "sentence_membership",
    "pool_sentence_graph",
    "symmetrize",
    "sparsify",
    "query_attention_per_sentence",
    "build_sentence_graph",
    # laplacian utilities
    "degree_vector",
    "eigh_laplacian",
    "induced_subgraph",
    "laplacian",
    "normalized_laplacian",
    "pseudoinverse",
    # H1 metrics
    "align_sentences_by_text",
    "davis_kahan_angle",
    "edge_rank_corr",
    "effective_resistance_drift",
    "effective_resistance_matrix",
    "laplacian_spectrum",
    "spectral_entropy",
    "spectrum_distance",
    "run_subgraph_hypothesis",
    "top_edge_recall",
    # NCut objective
    "cheeger_ratio",
    "cut_value",
    "jaccard",
    "ncut_q",
    "query_mass",
    "volume",
    # H2 solvers
    "anchored_spectral_cut",
    "exact_min_ncut",
    "fiedler_vector",
    "local_search",
    "query_anchored_spectral_cut",
    "random_cut",
    "spectral_cut",
    "top_query_attention_cut",
    # H2 entry point + diagnostics
    "algebraic_connectivity",
    "run_optimal_cut_hypothesis",
    # Streaming (memory-efficient) extraction
    "StreamedGraph",
    "StreamingExtractor",
]
