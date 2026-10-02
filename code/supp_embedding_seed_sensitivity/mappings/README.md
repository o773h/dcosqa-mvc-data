# Vertex-chain mappings for the embedding-seed-sensitivity experiment

The `raw/` directory contains 100 released logical-to-physical mappings:
20 graph instances, five regenerated embeddings per instance, and 150 logical
vertices per mapping.

The files document the physical embeddings used by the matched
embedding-seed-sensitivity outcomes. `code/validate_canonical_embeddings.py`
checks their coverage, seeds, graph degrees, physical-chain lengths, and
within-mapping physical-qubit uniqueness.

The raw mapping files contain only logical-node, physical-chain, chain-length,
and degree information. Normalized degree is computed from `Degree` when
needed; redundant centrality columns are not stored.

These mappings are linked only to the regenerated-embedding supplemental
experiment. The mappings for the canonical QA-Base--DC-OSQA comparison are
released separately under `data/metadata/canonical_embeddings/` and are used
by the vertex-level chain-length diagnostic in Supplementary Figure S5.
