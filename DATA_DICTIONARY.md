# Data dictionary

Column capitalization varies between recorded source tables and normalized
analysis tables. Scripts treat `Nodes`/`N` and `Instance`/`instance` as the
same fields.

## Candidate and graph fields

| Field | Definition |
|---|---|
| `N`, `Nodes` | Number of logical graph vertices |
| `instance`, `Instance` | Zero-based graph-instance identifier within an ensemble |
| `Bitstring` | Length-`N` binary candidate string; parse as text |
| `Occurrences` | Number of recorded reads represented by the row |
| `Size` | Number of selected vertices in the candidate cover |
| `IsValid`, `Valid` | Whether the candidate satisfies every vertex-cover constraint |
| `K_exact` | Independently certified minimum vertex-cover size for the graph |
| `strict_exact_verified` | Whether the candidate passes validity and exact-cardinality certification |
| `Energy` | Recorded QUBO or Ising energy, when retained |
| `Graph_Seed` | Seed used to generate the logical graph |
| `Edges` | Number of logical graph edges |
| `graph_hash_sha256` | SHA-256 identifier of the normalized graph edge set |

## Protocol and acquisition fields

| Field | Definition |
|---|---|
| `Solver`, `solver` | Recorded protocol label |
| `Solver_Canonical`, `solver_canonical` | Normalized protocol label |
| `OffsetScale`, `alpha` | Scale applied to the normalized anneal-offset score |
| `Num_Reads`, `Total_Read_Budget` | Requested or accumulated read budget |
| `Budget_Chunk_ID` | Acquisition chunk within an extended read budget |
| `Chain_Break_Frac` | Fraction of embedded logical chains broken in a returned sample |
| `EmbeddingRep` | Embedding replicate identifier |
| `EmbeddingSeed` | Seed used to generate the embedding |
| `Embedding_Source` | Recorded embedding identifier |
| `source_dataset`, `source_file` | Recorded source-dataset and file identifiers |

## Derived metric fields

| Field | Definition |
|---|---|
| `Unique_strict` | Number of distinct certified candidates observed |
| `Strict_occurrences` | Total occurrences of certified candidates |
| `Families_complete_linkage` | Number of complete-linkage Hamming families at the analysis threshold |
| `NormalizedEntropy` | Shannon entropy divided by the logarithm of observed support size |
| `ESSFraction` | Inverse-Simpson effective support divided by observed support size |
| `TopDecileShare` | Occurrence mass of the most frequent `ceil(0.1M)` certified candidates, where `M` is observed support size |
| `Top10Share` | Occurrence mass of the 10 most frequent certified candidates, or all candidates when `M < 10` |
| `GiniCoefficient` | Gini coefficient of certified-candidate occurrence probabilities |
| `zero_yield` | Whether no certified occurrence was observed in the condition |

Unless a script states otherwise, Hamming families use complete linkage with
distance threshold `0.1N`.

## Embedding-mapping fields

The canonical BA `m=2` mappings are stored in
`data/metadata/canonical_embeddings/ba_m2_vertex_chain_mappings.csv.gz`.
One mapping is shared by the matched QA-Base and DC-OSQA conditions for each
graph instance.

| Field | Definition |
|---|---|
| `Graph_Type`, `Graph_M` | Logical graph ensemble and BA attachment parameter |
| `Graph_Seed` | Seed used to generate the logical graph |
| `Embedding_Seed` | Seed supplied to the minor-embedding procedure |
| `LogicalNode` | Zero-based logical vertex identifier |
| `PhysicalChain` | JSON list of physical-qubit identifiers representing the logical vertex |
| `ChainLength` | Number of physical qubits in `PhysicalChain` |
| `Degree` | Degree of the logical vertex in the corresponding BA graph |
| `Mapping_SHA256` | Per-instance hash of the normalized logical-to-physical mapping, reported in the companion inventory |

The mapping tables intentionally omit degree- and betweenness-centrality
columns. Max-normalized degree is obtained directly as `Degree / max(Degree)`,
and betweenness centrality is a graph-derived protocol quantity rather than a
physical-chain field.

## Random 3-regular graph references

`code/supp_schedule_regular3_controls/reference/regular3_graph_registry.csv`
contains one row per graph instance with the graph seed, graph size, regular
degree, edge count, and graph hash. The companion
`regular3_graph_edges.csv` contains the normalized undirected edge set as
`Instance,u,v` rows.

For these graph references, `graph_hash_sha256` is computed from the UTF-8/ASCII
serialization

```text
N=150
u0,v0
u1,v1
...
```

where each edge is written as `min(u,v),max(u,v)` and the edge rows are sorted
lexicographically, with a line-feed terminator after every line. The references correspond to
`networkx.random_regular_graph(3, 150, seed=350000+Instance)` and were checked
against every recorded regular-3 candidate by recomputing feasibility and QUBO
energy.

## Protocol labels

| Recorded label | Canonical label |
|---|---|
| `QA-Base` | `QA-Base` |
| `DC-Offset(s=a)` | `DC-OSQA` at offset scale `a` |
| `BC-Offset(s=a)` | `BC-OSQA` at offset scale `a` |
| `DC-OSQA(s=a)` | Normalized DC-OSQA label retaining offset scale `a` |
| `BC-OSQA(s=a)` | Normalized BC-OSQA label retaining offset scale `a` |
| `SA-Max` | `SA-Max` |
| `SA-Recovery(Nr,Ms)` | SA recovery with `N` output samples and `M` sweeps |
| `Shuffled-DC(s=0.1)` | Shuffled assignment of the DC-OSQA offset multiset |

Blank fields indicate that a quantity was not recorded or is not applicable.
Boolean values are serialized as `True` or `False`. CSV files use UTF-8 text,
comma delimiters, one header row, and decimal points for numeric values.
