# Statement paragraph performance

## Investigation 1: repeated AST work and diagnostic emission

Status: **investigation only**. No production rule changes or new test files.
Rule SHA-256:
`8e51cb766f4c4ed72d17773584c1d3fa1fddcaf54a54cfd612792a28f944d068`.

Every runtime source matches the latest control-dispatch candidate's profile
snapshot. Only `root.zig`, which that driver does not import, differs. Reuse the
applicable latest complete-lint capture rather than confuse unrelated snapshots:
600 no-fix corpora after ten warmups, acknowledged perf windows, 12,193 samples,
zero lost samples and no missing callchains. Tail-call ownership uses that
binary's disassembled Linter return addresses. Inline source attribution uses
`addr2line` to distinguish the main loop's discovery, scratch, gap, and report
operations; phase attribution remains approximate.

### Measurement boundary

Zig 0.16.0 native ReleaseSafe, i7-13700HX CPU 8 P-core, sibling 9; single-threaded,
no lint-result cache. Same frozen corpus: 66 files, 1,114,088 bytes, 97,485 nodes,
197,393 tokens. The full linter emits 4,392 diagnostics and 3,995 proposed fixes;
this rule emits 3,149 diagnostics.

Fresh baseline calibration uses three separate processes, each with three warmups
and eight measured batches of four corpora. Below are medians of process medians,
not an A/B result or a latency improvement claim.

| Boundary | Cycles | Instructions | Wall time |
| --- | ---: | ---: | ---: |
| Prepared paragraphs | 11.477 M | 27.374 M | 3.520 ms |
| Prepared paragraphs + fix collection | 12.763 M | 31.721 M | 3.871 ms |
| Complete lint | 68.273 M | 165.033 M | 19.535 ms |

Prepared measurements exclude parsing/context prerequisites. Complete lint includes
parsing, context construction, all rules, diagnostics, allocation, and teardown.
Fix collection does not apply or format output. Sources are preloaded; this is
not a cold-disk/startup measurement. All PMU coverage readings are 100%; all nine
stage observations report thermal increases and sibling activity. Effective
frequency spans 3.26–3.53 GHz with no policy changes. Keep full-lint measurements
as the acceptance boundary for any subsequent optimization.

### Where the rule spends cycles

The rule owns **18.39% of complete-lint cycles**. Approximate disjoint phases:

| Phase | Whole lint | Within this rule |
| --- | ---: | ---: |
| Descriptions and reporting | 5.21% | 28.35% |
| Producer/guard/assert pairs | 2.41% | 13.12% |
| Gap boundaries and whitespace checks | 2.40% | 13.07% |
| Generic paragraph marking | 2.20% | 11.98% |
| Undefined-storage grouping | 2.05% | 11.17% |
| Block discovery and iteration | 1.62% | 8.82% |
| Other rule work / unattributed | 0.95% | 5.16% |
| Switch-arm paragraphs | 0.85% | 4.63% |
| Gap scratch allocation/initialization | 0.57% | 3.08% |
| Cleanup grouping | 0.11% | 0.63% |

Overlapping families, already included above:

- Diagnostics: 4.92% of whole lint / 26.75% of the rule.
- AST first/last-token helpers: 4.13% / 22.46%.
- Memory fills: 3.85% / 20.92%.
- Multiline queries: 2.11% / 11.46%.
- Category queries: 1.21% / 6.58%.
- Block decoding: 1.17% / 6.37%.
- Variable declaration decoding: 0.91% / 4.97%.
- Identifier spelling: 0.71% / 3.88%.

Memory-fill attribution is important: **3.35 of those 3.85 whole-lint percentage
points belong to reporting**, about 87% of this rule's sampled fills. Only 0.04
points belong to gap scratch initialization. Do not blame the small gap arrays
for the large fill total or redesign their allocator based on that aggregate.
Shared diagnostics are a separate optimization surface; this investigation does
not modify their API, ownership, or rendering.

### Untimed work census

Instrument a private rule copy without skipping any original decision, lookup,
or report. Per-file seen arrays count repeated category/multiline queries; those
arrays are instrumentation, not a proposed production cache. Rendered diagnostics
match the original on all 66 files in both Debug and ReleaseSafe. Instrumented
code is never used for timing or sampled attribution.

- 97,485 node visits; 2,614 blocks; 2,547 nonempty blocks.
- 938 nonempty blocks contain only one statement.
- 9,725 statement/gap slots. Gap and cleanup-flag payloads total 29,175 requested
  bytes across the corpus, in 5,094 allocation requests. This is not peak memory
  or complete rule allocation.
- 7,178 sibling gaps: 4,448 required, 663 forbidden, **2,067 unconstrained**.
- Undefined-storage classifier: 10,305 queries, 3,519 variable declarations,
  1,319 mutable declarations, 767 exact undefined-initializer hits. These are
  query/hit counts, including repeat queries, not distinct declaration counts.
- 8,355 multiline queries; **3,300 repeat a previously queried node** within
  the file (39.50%).
- 25,055 category queries; **15,659 repeated node queries** (62.50%).
- 234 single-name reference searches visit 1,322 tokens before completion;
  undefined-name map scans probe 2,395 identifier tokens. Name scanning is not
  the dominant target seen in the earlier top-down rule investigation.

### First experiments to run

1. **Gate block decoding with the scanned tag.** The main loop currently calls
   `blockStatements` for every node; only 2,614 of 97,485 nodes qualify. Keep all
   four block tags and existing block processing unchanged. This removes 97.32%
   of source-level discovery decoder requests, not 97.32% of rule cycles. The
   entire discovery phase is only 1.62% of whole lint, an upper bound on its
   complete deletion's benefit.
2. **Skip unconstrained gaps before resolving token boundaries.** The current
   loop resolves `firstToken`, `lastToken`, and `terminatedToken` even when the
   constraint immediately yields false. Moving that rejection earlier can omit
   2,067 pairs of boundary queries (4,134 first/last queries), without touching
   comment attachment, terminator handling, or constrained-gap semantics.
3. **Reject non-mutable declarations before full undefined-storage decoding.**
   Use supported variable-declaration tags and their main-token mutability to
   reject the 8,986 of 10,305 queries that are not mutable declarations. Preserve
   the current initializer, identifier-spelling, and name-token checks. Benchmark
   this separately: the extra gate has a cost, and other callers still decode
   variable declarations.

Only after those screens should a per-statement category/boundary cache be
considered. Repeated queries establish reuse, not a speedup. Eager facts could
compute unused boundaries for single-statement/unconstrained cases and enlarge
initialization; a lazy cache adds state and interface complexity. Preserve wrapper
and destructuring behavior, gap priority, undefined-storage diagnostics, cleanup
registrations, switch-arm rules, and exact fix bytes in any such experiment.

Do not simply skip single-statement blocks: a lone `var = undefined` can still
need a missing-use diagnostic, and same-line cleanup registration can still need
a line-break diagnostic/fix. Existing source comments and shared diagnostics must
remain intact. No cache, loop rewrite, or skip has been landed by this investigation.

Whole-rule deletion would bound speedup near 1.23×; eliminating all first/last-token
helper work beneath it would bound speedup near 1.04×. Neither is achievable by a
small gate, and overlapping ceilings must not be added. Compare isolated candidates
on exact outputs, then use balanced zigbench screens and confirmation at the
complete-lint boundary before making a retention or speedup claim.

Evidence:
`/tmp/experiments/voiced-lint/baseline-20260908/evidence/20260909T025406Z-paragraph-analysis`.
Retained artifacts include profile provenance and runtime-freshness checks, raw
stacks/reports/disassembly/source-line resolution, phase analysis, baseline JSON,
host conditions, the private instrumented rule, census driver, and Debug/ReleaseSafe
output-equivalence results. The original perf capture and matching executable remain
at the paths recorded in `provenance.json`.

## Experiment 2: block, unconstrained-gap, and mutable-declaration gates

Status: **all three retained** after independent screens, combined confirmation,
and a direct comparison against gap-only. The baseline is the exact rule hash
recorded above. Each arm freezes the same current implementation; only this rule
changes. No caches, allocation changes, new helpers, parser modifications, safety
relaxations, or parallel linting. The combined patch is +20/−5 lines, net +15.

### Implementation and correctness

- Gate block discovery on all four supported block tags before `blockStatements`.
  Empty/nonempty handling and single-statement processing remain intact.
- Skip `.unconstrained` gaps before resolving either boundary. Marking precedence,
  comment attachment, terminator handling, descriptions, and fix construction are
  unchanged for constrained gaps.
- Gate undefined-storage decoding on all four variable-declaration tags and the
  main token's `.keyword_var` tag. Zig 0.16's four variable decoders all initialize
  `mut_token` from `nodeMainToken`; this is the representation contract used by
  the early check. Initializer and exact name-spelling checks remain unchanged.

All independent variants and their combination preserve exact indexes, rendered
diagnostics, ordered edits, replacement bytes, fixed output, and applied/skipped
counts on **1,719 inputs in Debug and ReleaseSafe**. The set includes 240 new
private generated cases covering declaration representations, mutability,
initializers, quoted identifiers, gap layouts, attached comments, cleanup,
semantic pairs, nested scopes, and LF/CRLF. A tag census verifies all four block
and all four variable-declaration tags; global/aligned declarations are absent
from the main corpus but present in generated coverage.

The generator initially used uninitialized local declarations, which Zig's parser
rejects. Validation caught those malformed inputs rather than counting matching
parse errors as rule coverage. Six cases were corrected to valid nested extern
declarations, preserving the intended LF/CRLF variants; final validation was
rerun. Original failures and the invalid sample are retained. No production test
files or fixture snapshots were added or rewritten. Existing comments are intact.

### Independent screens

Same native Zig 0.16 ReleaseSafe / CPU 8 / single-threaded contract and frozen
66-file corpus as the investigation. Three warmups, eight measured batches of
four corpora per process; two balanced ABBA/BAAB blocks for each screen.

| Gate | Rule instructions | Rule cycles | Complete-lint instructions | Complete-lint cycles |
| --- | ---: | ---: | ---: | ---: |
| Block tags | −17.305% | −7.685% | −2.870% | −0.086% |
| Unconstrained gaps | −2.290% | −3.276% | −0.380% | −0.977% |
| Mutable declarations | −2.823% | −0.916% | −0.468% | +0.054% |

The block gate removes 4.737 M instructions per corpus, the gap gate 0.627 M,
and the mutable gate 0.772 M. Only the gap screen establishes a promising
complete-lint cycle trend on its own. Block and mutable checks should not be
advertised as standalone whole-lint latency wins based on these screens.

### Combined confirmation

Eight predeclared balanced four-process blocks, with the same work/batch contract.
Percentages use geometric within-block B/A ratios; intervals are exploratory 95%
bootstraps over blocks, not independent batches. All PMU coverage readings are
100%. Unfavorable and heavily disturbed blocks remain in the analysis.

| Boundary | Instructions | Cycles (interval) | Wall time (interval) |
| --- | ---: | ---: | ---: |
| Prepared paragraphs | −22.543% | −13.193% (−20.059%, −6.350%) | −11.549% (−22.351%, −2.429%) |
| Complete lint | −3.739% | −3.698% (−8.600%, −0.380%) | −4.327% (−10.106%, −0.053%) |
| Complete lint + fixes | −3.522% | −1.766% (−2.405%, −1.070%) | −1.986% (−4.405%, −0.066%) |
| Directory lint | −3.661% | +1.188% (−1.668%, +6.235%) | +5.727% (−1.387%, +18.843%) |

Raw complete-lint medians: **67.984 M → 66.653 M cycles** (about 2% lower),
**165.034 M → 158.863 M instructions**, and **16.690 → 16.497 ms**. Prepared-rule
cycle medians are 11.482 M → 9.934 M. The combined no-fix instruction reduction
is 6.171 M per corpus. These raw medians are not the paired estimators above.

The paired plain-lint estimate is strongly influenced by a disturbed block;
do not present −3.7% as a precise universal cycle/latency improvement. Directory
improvement versus the original baseline is **not established**. All 128 stage
observations report thermal increases, 127 report sibling activity, frequency
spans 2.99–4.27 GHz, and no frequency-policy changes occur. This is warm corpus
linting, not disk-cold or startup performance. Fix-collecting measurements still
exclude application/formatting.

### Are the other two gates worthwhile beyond gap-only?

A second eight-block confirmation compares **gap-only (A)** with **all three (B)**.
This isolates the joint incremental benefit of block/mutability gating; it does
not identify a separate speedup for each gate or justify adding estimates from
different comparisons.

| Boundary | Additional instructions | Additional cycles (interval) | Wall time (interval) |
| --- | ---: | ---: | ---: |
| Prepared paragraphs | −20.728% | −11.643% (−15.104%, −9.446%) | −6.447% (−11.748%, −0.414%) |
| Complete lint | −3.372% | −0.544% (−0.768%, −0.296%) | +4.999% (−0.797%, +15.287%) |
| Complete lint + fixes | −3.172% | −0.617% (−0.756%, −0.462%) | +4.574% (−0.971%, +14.282%) |
| Directory lint | −3.301% | −0.761% (−0.836%, −0.667%) | −1.892% (−3.551%, −0.476%) |

This supports retaining the two additional small gates as a joint CPU-work
improvement, rather than relying only on their isolated instruction reduction.
It does **not** establish an additional preloaded-lint wall-time improvement.
All coverage readings are 100%; 128/128 observations report thermal increases,
124 report sibling activity, frequency spans 3.51–4.33 GHz, and there are no
policy changes. Branch misses increase about 1% in complete lint, so fewer
instructions are not interchangeable with fewer cycles or lower elapsed time.

### Generated code, integration, and reprofile

Native disassembly confirms all three rejection branches precede the expensive
work: a block-tag range test gates AST-copy setup and block decoding; a constraint
byte test skips boundary setup and both first/last-token calls; the variable-tag
and keyword tests return before `fullVarDecl`. Successful candidates still use
standard decoders and retain runtime safety checks.

Before/after profiles each cover 600 complete no-fix corpora after ten warmups,
with acknowledged perf windows. Captures contain 9,752 / 9,572 samples and zero
lost samples. Each uses its own disassembled Linter return addresses for tail-call
ownership. Sampled paragraph share falls **18.85% → 16.40%**. Other disjoint shares
are parsing 44.63% → 46.36%, top-down 13.08% → 13.27%, and control flow
11.28% → 11.13%. Whole-lint boundary-helper share is 6.54% → 6.03%, overlapping
those owners. Relative shares are attribution, not another latency estimate or
evidence of regressions in untouched components.

The integrated rule matches the measured combined source byte-for-byte. Full
project tests pass in Debug and ReleaseSafe, along with rule/native formatting
and the ReleaseSafe install build, all through `agent-run`. Final generated-input
comparisons also pass in both modes. Unrelated concurrent work is preserved.

Evidence:
`/tmp/experiments/voiced-lint/baseline-20260908/evidence/20260909T031603Z-paragraph-gates`.
Raw schedules, invocations, counters, summaries, host conditions, source/binary
hashes, patches, valid/invalid generated inputs, validators, tag census, profiles,
disassembly, scripts, and check records are retained. Frozen implementations live
under the same experiment base in `paragraph-gates-{base,block,gap,mutable,combined}`.
