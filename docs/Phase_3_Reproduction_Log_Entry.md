# Reproduction Log Entry - Phase 3: Heterogeneous Graph Reconstruction

**Completion date:** 2026-08-19  
**Status:** COMPLETE  
**Previous phase:** Phase 2 - Temporal Reconstruction  
**Next phase:** Phase 4 - Model Reconstruction

## Phase objective

Reconstruct the ITRS structural heterogeneous graph from the relationships genuinely supported by the available Crunchbase exports, freeze role-based graph identity, prevent T60 structural leakage, materialize scientific node/edge tables, and produce deterministic model-ready integer encodings.

## Chronological subphase record

| Subphase | Objective | Main result / decision |
|---|---|---|
| 3.1 | Recover exact ITRS graph-construction principles from the paper | Final graph uses Investor/Startup role nodes; Company/Human are relationship-generating intermediaries; investment events stay separate; exact 103-relation vocabulary unrecoverable |
| 3.2.1 | Raw dataset/schema inventory | Initial raw Companies/Investor/Funding inventory established |
| 3.2.2 | Test Companies as universal master registry | Companies covers 100% of canonical Startups but only 25.7756% of canonical Investors; Investor file covers 100% of canonical Investors |
| 3.2.3 | Cross-role UUID identity audit | 9,633 canonical UUIDs occur in both Investor and Startup roles; shared IDs represent same underlying entities; role-namespaced graph IDs required |
| 3.2.4 | Identity anomaly closure | Company chunk overlap confirmed as export overlap; 110 self-ID canonical investment events preserved; UUID remains identity key |
| 3.2.5 | Relationship-source discovery | Auxiliary Acquisitions, People, Hubs, Schools, Events, Contacts added to source universe; no automatic node-type expansion |
| 3.2.6 | Relationship-source schema/semantics audit | Founder URL bridge and acquisitions identified as highest-value relation sources; contacts/hubs/events rejected or deferred |
| 3.2.7 | Cross-dataset reference bridge audit | `companies.founder_links -> people.link` resolves 99.9969%; Contacts exact ID bridge fails; Schools deferred |
| 3.2.8 | Acquisition endpoint resolution | 158,307 rows resolve both endpoints; conservative accepted statuses and pre-T60 structural cutoff defined |
| 3.2.9 | Conservative acquisition candidate audit | 23,301 unique leakage-safe directed underlying acquisition pairs; 32,000 role-specific forward candidates |
| 3.2.10 | Shared-founder dedup/leakage audit | 47,409 final leakage-safe symmetric role-pair candidates; 94,818 directed records if stored bidirectionally |
| 3.2.11 | Relation support matrix / audit closure | Two supported structural sources carried forward: `SHARED_FOUNDER`, `ACQUIRED`; all other sources classified explicitly |
| 3.3.1 | Freeze graph schema | 2 node types, 3 base semantic relations, 12 typed channels, 158,818 expected directed edges |
| 3.3.2 | Materialize node table | 477,564 role nodes; 467,931 underlying entities; 100% registry coverage; no identity failures |
| 3.3.3 | Materialize edge table | 158,818 edges; all integrity checks PASS; zero T60 structural leakage |
| 3.4.1 | Deterministic model-ready encoding | Node indices 0-477,563; relation IDs 0-11; `edge_index` shape (2, 158,818); zero round-trip mismatches |
| 3.4.2 | Structural coverage audit | 74,757 connected nodes; founder/acquisition sources complementary; 72.1682% of T60 pairs have neither endpoint structurally connected |
| 3.4.3 | Final integrity / closure | All closure checks PASS; closure manifest written; Phase 3 frozen |

## Final graph specification

```text
Node types:
  investor
  startup

Semantic relations:
  shared_founder
  acquired
  acquired_by

Typed relation channels:
  12

Role nodes:
  477,564

Directed structural edge records:
  158,818

Historical investment-event structural edges:
  0

T60 structural leakage edge records:
  0
```

## Identity policy

- Underlying identity key: Crunchbase UUID.
- Graph identity key: semantic role + UUID.
- Node IDs: `investor::<uuid>` and `startup::<uuid>`.
- Dual-role underlying entities: 9,633.
- Dual-role graph nodes: 19,266.
- Never merge dual-role role nodes during model reconstruction.

## Relation policy

### SHARED_FOUNDER

- Construction: `Company -> Person <- Company` through exact Crunchbase Person URLs.
- Exact founder URL resolution: 99.9969%.
- Final symmetric candidates: 47,409.
- Materialized directed records: 94,818.
- Temporal provenance: `current_snapshot_unversioned`.

### ACQUIRED / ACQUIRED_BY

- Conservative endpoint evidence only.
- Structural cutoff: `announced_on < 2026-01-01`.
- Final unique directed underlying pairs: 23,301.
- Forward role-projected `ACQUIRED` edges: 32,000.
- Explicit inverse `ACQUIRED_BY` edges: 32,000.
- Temporal provenance: `timestamped_pre_T60`.
- Relation is a Crunchbase-specific adaptation of the ITRS structural-relation principle.

## Leakage policy

The T60 holdout contains 22,327 unique directed Investor-Startup pairs and 22,326 unique unordered underlying entity pairs.

For every direct structural relationship:

```text
if unordered(underlying_src_id, underlying_dst_id)
   matches a T60 held-out underlying pair:
       exclude the structural relationship
```

Final structural leakage edge records: **0**.

## Final graph variants

| Variant | Edge records |
|---|---:|
| Core | 158,818 |
| Founder-only ablation | 94,818 |
| Acquisition-only ablation | 64,000 |

## Structural coverage record

| Population | Connected | Total | Connected % |
|---|---:|---:|---:|
| Investors | 8,771 | 165,975 | 5.2845% |
| Startups | 65,986 | 311,589 | 21.1773% |
| All role nodes | 74,757 | 477,564 | 15.6538% |

Source overlap:

- Founder + acquisition: 10,543 nodes
- Founder only: 39,771 nodes
- Acquisition only: 24,443 nodes
- Neither: 402,807 nodes

T60 core pair coverage:

- Both endpoints connected: 653 = 2.9247%
- Investor only connected: 1,798 = 8.0530%
- Startup only connected: 3,763 = 16.8540%
- Neither endpoint connected: 16,113 = 72.1682%

**Decision:** retain all structural isolates and keep the frozen T60 evaluation universe unchanged. Structural coverage may be used later only as an evaluation diagnostic slice.

## Final scientific artifacts

```text
data/experimental/phase_3/graph/nodes.parquet
data/experimental/phase_3/graph/edges.parquet
```

## Final model-ready artifacts

```text
data/experimental/phase_3/model_ready/node_index.parquet
data/experimental/phase_3/model_ready/relation_index.csv
data/experimental/phase_3/model_ready/edge_index.npy
data/experimental/phase_3/model_ready/edge_type.npy
data/experimental/phase_3/model_ready/edge_manifest.parquet
data/experimental/phase_3/model_ready/graph_variant_masks.npz
data/experimental/phase_3/model_ready/model_ready_graph_metadata.json
data/experimental/phase_3/model_ready/node_structural_coverage.parquet
data/experimental/phase_3/model_ready/phase_3_closure_manifest.json
```

## Known limitations recorded at closure

1. Full ITRS 103-relation vocabulary and preprocessing rules are not published.
2. Most published Tianyancha primitive relation families are unavailable in the Crunchbase exports.
3. `SHARED_FOUNDER` has current-snapshot rather than historical relation-time provenance.
4. `ACQUIRED` is a Crunchbase-specific structural adaptation.
5. The structural graph is sparse: 402,807 role nodes are isolated.
6. Structural coverage is limited for T60 prediction pairs.

## Rejected alternatives

- Do not force exactly 103 relation types.
- Do not invent Tianyancha-equivalent shareholder/client/supplier/competitor relations from unrelated Crunchbase attributes.
- Do not use categories or locations as strict structural relations.
- Do not convert funding/investment interactions into structural R-GCN edges.
- Do not merge Investor/Startup role nodes sharing one underlying UUID.
- Do not filter out graph isolates.
- Do not filter T60 evaluation cases by structural coverage.
- Do not accept ambiguous acquisition endpoint resolutions arbitrarily.

## Questions handed to Phase 4

Phase 4 should reconstruct the ITRS model architecture against the frozen graph and temporal interaction artifacts. The graph itself should not be reopened. Remaining model questions include exact implementation of R-GCN layers/bases, description/label feature construction, trend extraction and GRU sequence encoding, scoring network, negative sampling, training batches, and evaluation candidate construction according to the paper and the available Crunchbase data.

## Suggested commit

```text
docs: close Phase 3 heterogeneous graph reconstruction and freeze model-ready graph
```
