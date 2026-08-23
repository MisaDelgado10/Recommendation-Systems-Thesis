# Phase 3 - Heterogeneous Graph Reconstruction ✅

**Completion date:** 2026-08-19  
**Project:** ITRS Crunchbase Reproduction  
**Status:** COMPLETE  
**Next phase:** Phase 4 - Model Reconstruction

## Objective

Reconstruct the ITRS heterogeneous structural graph using only relationships that are defensibly observable in the available Crunchbase exports, while preserving the immutable Phase-1 investment-event layer and the frozen Phase-2 temporal split.

Phase 3 deliberately separates the **static structural graph** from the **temporal investment-event sequence**. Historical investment events are therefore not inserted as R-GCN structural edges.

## Paper-grounded graph interpretation

The ITRS paper distinguishes Organization, Brand, Company, and Human concepts. Organization/Brand become the final recommendation graph nodes, while Company/Human can act as intermediary relationship sources. The paper reports ten primitive Tianyancha relationship families and a final graph with 103 relationships, but it does not publish the complete 103-relation vocabulary or all preprocessing rules.

**Reproduction rule:** reproduce the disclosed relationship-construction principle with defensible Crunchbase sources; do not force 103 relations or invent unavailable Tianyancha equivalents.

## Final graph identity

Graph node types:

```text
investor
startup
```

Graph IDs are role-namespaced:

```text
investor::<crunchbase_uuid>
startup::<crunchbase_uuid>
```

The raw Crunchbase UUID is preserved as `raw_entity_id`.

Final identity counts:

| Metric | Value |
|---|---:|
| Investor role nodes | 165,975 |
| Startup role nodes | 311,589 |
| Total role nodes | 477,564 |
| Unique underlying entities | 467,931 |
| Dual-role underlying entities | 9,633 |
| Dual-role graph nodes | 19,266 |
| Duplicate graph node IDs | 0 |
| Missing display names | 0 |

Both authoritative registries have 100% canonical-role coverage. All 9,633 dual-role identities have matching display names across the Investor and Company registries.

## Available relationship-source decisions

| Source / candidate | Status | Phase-3 decision |
|---|---|---|
| `companies.founder_links -> people.link` / `SHARED_FOUNDER` | Supported | Core-graph candidate |
| Acquisitions / `ACQUIRED` | Supported Crunchbase adaptation | Core-graph candidate |
| Funding / `INVESTMENT_EVENT` | Separate temporal layer | Do not use as structural R-GCN edge |
| People schools / `SHARED_SCHOOL` | Deferred | Exclude from initial graph |
| Contacts | Excluded strict | No usable exact person/org bridge |
| Hubs | Unsupported | Exclude |
| Events | Unsupported | Exclude |
| Categories / category groups | Extension only | Feature layer; exclude from initial graph |
| Location attributes | Extension only | Feature layer; exclude from initial graph |

## SHARED_FOUNDER audit

Exact bridge:

```text
companies.founder_links -> people.link
```

Resolution:

- Unique founder URLs: 319,284
- Exactly resolved to one People ID: 319,274
- Ambiguous links: 7
- Unresolved links: 3
- Exact resolution: **99.9969%**

Filtering:

| Stage | Count |
|---|---:|
| Founder-induced pair occurrences | 66,227 |
| Unique role-node pairs | 54,332 |
| After removing same-underlying-entity pairs | 47,431 |
| After removing T60 holdout pairs | **47,409** |

Final symmetric role pairs:

- Startup--Startup: 38,411 = 81.0205%
- Investor--Startup: 7,768 = 16.3851%
- Investor--Investor: 1,230 = 2.5944%

Materialized in both directions:

```text
94,818 directed shared_founder edge records
```

Temporal provenance:

```text
current_snapshot_unversioned
```

## ACQUIRED audit

Accepted endpoint evidence statuses:

```text
name_website_agree
ambiguous_name_resolved_by_website
unique_name_only
```

Filtering:

| Stage | Rows remaining |
|---|---:|
| Fully resolved input | 158,307 |
| Accepted endpoint evidence | 157,989 |
| Observable before T60 | 152,212 |
| Remove self-acquisitions | 152,211 |
| Both endpoints in canonical roles | 23,329 |
| Remove T60 holdout underlying pairs | **23,306** |
| Unique directed underlying pairs | **23,301** |

Temporal policy:

```text
announced_on < 2026-01-01
```

Role-specific forward projection:

| Source | Target | ACQUIRED edges |
|---|---|---:|
| Investor | Investor | 1,565 |
| Investor | Startup | 11,427 |
| Startup | Investor | 1,271 |
| Startup | Startup | 17,737 |
| **Total** | | **32,000** |

An explicit `ACQUIRED_BY` inverse is materialized for every forward edge, producing another 32,000 records.

`ACQUIRED` is a **Crunchbase-specific adaptation** following the ITRS company-relationship construction principle; it is not claimed to be one of the paper's ten published primitive Tianyancha relation labels.

## Final graph schema

Base semantic relations:

```text
shared_founder
acquired
acquired_by
```

Typed relation key:

```text
src_type|relation|dst_type
```

Final counts:

| Metric | Value |
|---|---:|
| Graph node types | 2 |
| Role nodes | 477,564 |
| Base semantic relations | 3 |
| Typed R-GCN relation channels | 12 |
| `shared_founder` directed edges | 94,818 |
| `acquired` forward edges | 32,000 |
| `acquired_by` inverse edges | 32,000 |
| **Core directed structural edges** | **158,818** |
| Historical investment structural edges | 0 |
| T60 structural leakage edges | 0 |

## Final scientific graph

```text
data/experimental/phase_3/graph/
├── nodes.parquet    # 477,564 rows
└── edges.parquet    # 158,818 rows
```

`nodes.parquet` columns:

```text
node_id
node_type
raw_entity_id
display_name
source_registry
underlying_entity_has_dual_role
```

`edges.parquet` stores role-namespaced endpoints, source/target type, base relation, typed relation key, underlying UUIDs, provenance, temporal provenance, observation dates where available, support count, and inverse-edge flag.

## Structural graph integrity

All Phase-3.3 edge checks passed:

- Structural edge rows: 158,818
- Base relations: 3
- Typed channels: 12
- Duplicate edge IDs: 0
- Duplicate `(src, relation, dst)` triples: 0
- Graph-node self loops: 0
- Same-underlying-entity structural relations: 0
- Missing source/destination nodes: 0
- Endpoint type/UUID mismatches: 0
- Missing `SHARED_FOUNDER` reverse records: 0
- Missing `ACQUIRED_BY` inverse records: 0
- T60 holdout leaking edge records: **0**

## Model-ready graph encoding

```text
data/experimental/phase_3/model_ready/
├── node_index.parquet
├── relation_index.csv
├── edge_index.npy
├── edge_type.npy
├── edge_manifest.parquet
├── graph_variant_masks.npz
├── model_ready_graph_metadata.json
├── node_structural_coverage.parquet
└── phase_3_closure_manifest.json
```

Deterministic node indices:

```text
Investor: 0 ... 165974
Startup:  165975 ... 477563
```

Deterministic relation IDs:

```text
0 ... 11
```

Array shapes:

```text
edge_index.shape = (2, 158818)
edge_index.dtype = int64
edge_type.shape  = (158818,)
edge_type.dtype  = int64
```

All source-node, destination-node, and relation round-trip mismatch counts are 0.

## Final typed relation vocabulary

| ID | Source | Relation | Target | Edges |
|---:|---|---|---|---:|
| 0 | investor | acquired_by | investor | 1,565 |
| 1 | investor | acquired_by | startup | 1,271 |
| 2 | investor | acquired | investor | 1,565 |
| 3 | investor | acquired | startup | 11,427 |
| 4 | investor | shared_founder | investor | 2,460 |
| 5 | investor | shared_founder | startup | 7,768 |
| 6 | startup | acquired_by | investor | 11,427 |
| 7 | startup | acquired_by | startup | 17,737 |
| 8 | startup | acquired | investor | 1,271 |
| 9 | startup | acquired | startup | 17,737 |
| 10 | startup | shared_founder | investor | 7,768 |
| 11 | startup | shared_founder | startup | 76,822 |

## Graph variants

| Variant | Relations | Directed edges |
|---|---|---:|
| Core | `shared_founder`, `acquired`, `acquired_by` | **158,818** |
| Founder-only ablation | `shared_founder` | 94,818 |
| Acquisition-only ablation | `acquired`, `acquired_by` | 64,000 |

Use `graph_variant_masks.npz`; do not rebuild variants with different graph semantics.

## Structural coverage

Core graph:

| Node type | Connected | Total | Connected share |
|---|---:|---:|---:|
| Investor | 8,771 | 165,975 | 5.2845% |
| Startup | 65,986 | 311,589 | 21.1773% |
| ALL | **74,757** | **477,564** | **15.6538%** |

Source complementarity:

| Coverage class | Nodes |
|---|---:|
| Founder + acquisition | 10,543 |
| Founder only | 39,771 |
| Acquisition only | 24,443 |
| Neither | 402,807 |

Acquisition relations therefore add 24,443 structurally covered nodes that would not be reached by `SHARED_FOUNDER` alone.

## T60 structural coverage

Unique T60 endpoints under the core graph:

- Investors: 864 / 11,884 connected = 7.2703%
- Startups: 1,349 / 8,992 connected = 15.0022%

T60 pair coverage:

| Pair state | Pairs | Share |
|---|---:|---:|
| Both endpoints connected | 653 | 2.9247% |
| Investor only connected | 1,798 | 8.0530% |
| Startup only connected | 3,763 | 16.8540% |
| Neither endpoint connected | 16,113 | 72.1682% |
| At least one endpoint connected | 6,214 | 27.8318% |

**Do not filter T60 by structural coverage.** `node_structural_coverage.parquet` is retained for later diagnostic evaluation slices only.

## Global T60 structural leakage policy

The frozen Phase-2 holdout contains 22,327 unique directed Investor-Startup pairs, corresponding to 22,326 unique unordered underlying entity pairs because one underlying pair occurs in both role directions.

For every structural relation:

```text
if unordered(underlying_src_id, underlying_dst_id)
   is a T60 held-out underlying pair:
       remove the direct structural relation
```

Final leakage result:

```text
T60 structural leaking edge records = 0
```

## Known limitations

1. **Full ITRS relation vocabulary unavailable.** The paper reports 103 final relationships but does not publish the full vocabulary or preprocessing rules.
2. **Most Tianyancha primitive relations unavailable.** The available Crunchbase exports do not expose entity-level shareholder, controlling, supplier, client, competitor, or historical shareholding relationships.
3. **Founder relationships are temporally unversioned.** `SHARED_FOUNDER` is current-snapshot side information and may not represent a historically observable relation at every prediction time.
4. **Acquisition is a Crunchbase adaptation.** It is not one of the ten explicitly published primitive Tianyancha relation labels.
5. **Structural graph sparsity.** 402,807 of 477,564 role nodes are structurally isolated.
6. **Limited structural information for T60 pairs.** 72.1682% of held-out T60 pairs have no structural neighbor on either endpoint.

## Confirmed Phase-3 scripts

```text
scripts/phase_3_2_1_raw_dataset_schema_inventory.py
scripts/phase_3_2_2_companies_master_entity_audit.py
scripts/phase_3_2_3_cross_role_identity_and_company_duplicate_audit.py
scripts/phase_3_2_4_identity_anomaly_closure_audit.py
scripts/phase_3_2_5_relationship_source_dataset_discovery.py
scripts/phase_3_2_6_relationship_source_schema_semantics_audit.py
scripts/phase_3_2_7_cross_dataset_reference_bridge_audit.py
scripts/phase_3_2_8_acquisition_endpoint_resolution_audit.py
scripts/phase_3_2_9_conservative_acquisition_acceptance_audit.py
scripts/phase_3_2_10_shared_founder_candidate_leakage_audit.py
scripts/phase_3_2_11_relation_support_matrix_closure.py
scripts/phase_3_3_1_graph_schema_specification.py
scripts/phase_3_3_2_graph_node_table_materialization.py
scripts/phase_3_3_3_structural_edge_table_materialization.py
scripts/phase_3_4_1_model_ready_graph_encoding.py
scripts/phase_3_4_2_structural_coverage_relation_degree_audit.py
scripts/phase_3_4_3_final_integrity_and_closure.py
```

## Versionable configuration and compact artifacts

Version:

```text
configs/phase_3_itrs_graph_schema.json
scripts/phase_3_*.py
docs/reproduction_log.md
README.md
small audit/config CSV/JSON summaries where useful
```

Generally keep outside Git:

```text
data/raw/*.csv
large Parquet node/edge artifacts
NumPy tensor artifacts
large diagnostic outputs
```

## Phase-4 handoff constraints

Phase 4 must:

1. Treat Phase-3 graph semantics as frozen.
2. Keep all 477,564 role nodes, including structural isolates.
3. Preserve role-namespaced node identity.
4. Use `relation_index.csv` as the authoritative typed-relation mapping.
5. Use `edge_index.npy` / `edge_type.npy` as the authoritative model-ready structural arrays.
6. Keep historical investment interactions separate as the Phase-2 temporal interaction sequence.
7. Preserve the zero-leakage T60 structural mask.
8. Preserve acquisition cutoff `announced_on < 2026-01-01`.
9. Preserve founder provenance `current_snapshot_unversioned`.
10. Use existing graph-variant masks for ablations instead of changing the graph construction pipeline.
11. Use structural-coverage fields only for diagnostic evaluation slices, never for primary test filtering.

## Next

**Phase 4 - Model Reconstruction**

Start from the frozen Phase-3 scientific graph and model-ready arrays. Do not reopen graph construction unless a separately named extension experiment is explicitly introduced.

## Suggested commit

```text
docs: close Phase 3 heterogeneous graph reconstruction and freeze model-ready graph
```
