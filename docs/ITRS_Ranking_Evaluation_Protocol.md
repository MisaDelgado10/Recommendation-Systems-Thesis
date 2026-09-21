# ITRS Ranking Evaluation Protocol - What "Top 10" Means

**Scope:** Frozen Phase-5/Phase-6 evaluation protocol  
**Applies to:** T60 validation and final test  
**Metrics:** HR@10 and NDCG@10

---

## 1. Evaluation unit

The evaluation unit is one held-out T60 investment event.

For a case with investor `o` and the actually observed startup `b+`, the evaluation system creates exactly 100 candidates:

```text
1 observed positive startup b+
99 frozen sampled negative startups
-----------------------------------
100 candidates
```

> "Top 10" means top 10 within this 100-candidate sampled ranking task. It does not mean top 10 among all 311,589 startup nodes.

---

## 2. What score determines the ranking?

The model outputs one raw logit for every investor-startup candidate pair.

The frozen representation is:

```text
Investor R_o:
[F_t, L_o, F_d,o, F_s,o] = 160 dimensions

Startup R_b:
[L_b, F_d,b, F_s,b] = 120 dimensions

Pair:
[R_o, R_b] = 280 dimensions

Scoring MLP:
280 -> 128 -> 64 -> 32 -> 16 -> 1
```

Where `F_t` is trend, `L` is the latent embedding, `F_d` is description/category information, and `F_s` is the structural graph representation.

No manually defined VC score, startup-quality score, valuation score, or future-success score determines Top 10.

---

## 3. Exact ranking rule

For each case, the 100 candidates are sorted by:

```text
1. raw model logit descending
2. startup_local ascending as deterministic tie-break
```

The observed positive startup receives a 1-based rank.

Therefore:

```text
positive startup is in Top 10
<=> positive_rank <= 10
```

The sigmoid probability is not required for ranking because sigmoid is monotonic for finite logits.

---

## 4. HR@10

Per event:

```text
HR@10 = 1 if positive_rank <= 10
HR@10 = 0 otherwise
```

The split-level metric is the arithmetic mean across events.

Frozen final test:

```text
Cases:        20,264
Hits@10:      11,072
HR@10:        0.546387682590
```

Interpretation:

> In about 54.64% of the sampled 100-candidate test cases, the actually observed startup was ranked in the first ten positions.

---

## 5. NDCG@10

Because each case has exactly one relevant startup:

```text
NDCG@10 =
    1 / log2(positive_rank + 1), if positive_rank <= 10
    0,                            otherwise
```

Examples:

| Rank | HR@10 | NDCG@10 |
|---:|---:|---:|
| 1 | 1 | 1.0000 |
| 2 | 1 | 0.6309 |
| 3 | 1 | 0.5000 |
| 5 | 1 | 0.3869 |
| 10 | 1 | 0.2891 |
| 11 | 0 | 0 |
| 50 | 0 | 0 |

Frozen final-test NDCG@10:

```text
0.398709730124
```

---

## 6. What counts as the correct startup?

The positive is the startup actually observed in the held-out T60 investment event for the focal investor.

Thus the evaluation asks:

> Can the model rank the observed investment target above sampled alternatives?

It does not directly test whether the startup is objectively good, will succeed, has the highest valuation, or should receive investment.

---

## 7. Negative candidates

The frozen protocol samples 99 startup-role nodes while excluding:

- the focal positive;
- startups that were already positive pairs for the investor before T60.

Other simultaneous T60 positives are not automatically excluded because they were not historical information.

Candidate lists are generated once and reused. They are never resampled after model scores are observed.

---

## 8. Final evaluation result

The selected checkpoint was display Epoch 19, chosen from validation only.

```text
Test cases:            20,264
Candidates / case:     100
HR@10:                 0.546387682590
NDCG@10:               0.398709730124
Hits@10:               11,072
Mean positive rank:    23.034347
Median positive rank:  7
```

No optimizer or backward pass was used in final evaluation, and the checkpoint was not reselected after seeing test.

---

## 9. Critical caveat

A sampled 100-candidate HR@10 is not a full-catalog ranking metric.

Therefore `HR@10 = 0.5464` must not be reported as "the correct startup is in the top 10 out of all startups 54.64% of the time."

It means top 10 out of the fixed 100-candidate sampled evaluation set.
