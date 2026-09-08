# Top-down declaration performance

## Measurement contract

- Zig 0.16.0, native ReleaseSafe; Intel i7-13700HX, pinned P-core (logical CPU 8).
- Original rule snapshot SHA-256 starts `cee8116df0a17ad3`; subsequent concurrent
  rule changes are preserved and checked separately before integration.
- Frozen repository corpus: 66 Zig files, 1,114,088 bytes, 97,485 AST nodes,
  197,393 tokens; six rules, 4,489 diagnostics and 4,099 proposed fixes.
- One operation lints the entire corpus. Sources are preloaded for `lint_file`;
  directory measurements include traversal and reads, with the lint cache disabled.
- zigbench hardware counters cover measured batches only. Wall times are descriptive:
  the host reports throttle events and occasional SMT-sibling activity.
- A separate symbolized driver profiles 300 complete lint operations after preparation
  and ten warmups. Acknowledged perf enable/disable commands exclude setup and teardown.
  No lost samples in the successful baseline profile.

## Baseline and headroom

Fixed-work repeat, median per corpus, ReleaseSafe:

| Boundary | Wall time | Cycles | Instructions |
| --- | ---: | ---: | ---: |
| AST construction | 8.87 ms | 32.22 M | 91.44 M |
| Prepared rules | 15.53 ms | 54.80 M | 145.48 M |
| Complete lint | 25.17 ms | 89.88 M | 238.44 M |
| Directory lint | 25.77 ms | 90.40 M | 241.99 M |

Prepared phase timings are not an exact additive decomposition: storage and cache
lifetimes differ. Complete-lint cycle samples attribute 33.55% to parsing, 22.66%
to top-down declarations, 17.26% to paragraphs and 13.73% to control flow.
`Ast.tokenSlice` accounts for 12.03% inclusive, overlapping those callers; variable
spelling tokens are re-tokenized. Removing all top-down work would bound the speedup
at approximately 1.29×, not predict an achievable result.

## Experiment 1: omit empty function scans

Hypothesis: `lintFunctionOrder` needlessly scans container tokens and re-tokenizes
identifiers when its function table is empty. Returning before that scan preserves
all diagnostics while reducing retired instructions and cycles. Its entire baseline
region accounts for 16.60% of cycles (absolute ceiling 1.20×); only the empty-table
subset is removable, so the actual ceiling is lower.

Plan: independent frozen baseline/candidate binaries; exact rendered diagnostics and
fix-output comparison outside timing; two balanced four-process blocks for screening.
If promising, eight predeclared ABBA/BAAB blocks, fixed work, fresh process per arm,
including complete lint and directory lint. Do not pool batches as independent runs.

### Confirmation

Eight balanced four-process blocks; 16 measured batches per process, four complete
corpora per batch, three warmups. All PMU coverage readings were 100%. Percent changes
below use the geometric mean of within-block B/A ratios, not pooled batch observations.
Intervals are exploratory 95% bootstrap intervals over the eight process blocks.

| Boundary | Instructions | Cycles (interval) | Wall time (interval) |
| --- | ---: | ---: | ---: |
| Top-down rule | −4.05% | −2.51% (−6.77%, +3.14%) | −2.08% (−7.90%, +5.08%) |
| Complete lint | −0.98% | −1.83% (−3.64%, +0.29%) | −1.49% (−3.10%, +0.24%) |
| Complete lint + fixes | −0.94% | −2.61% (−4.02%, −1.45%) | −2.38% (−3.93%, −0.80%) |
| Directory lint | −0.96% | −0.60% (−1.49%, +0.46%) | −0.72% (−2.05%, +0.59%) |

The candidate removes about 2.335 million instructions per corpus in every boundary.
Branch misses also decrease. That supports the deletion mechanism; plain/directory
latency improvement is **not established** on this contended host. The fix-collecting
boundary improved in this run; do not generalize that to a guaranteed overall speedup.

Decision: retain the narrowly scoped empty-table guard as removal of demonstrably
unnecessary work, not as a broad throughput claim. Net production change: +5 lines;
no extra allocation, metadata columns, helpers, or visitor machinery. All existing
comments remain intact. Timing experiments retain runtime safety.

Correctness: compare rendered diagnostics, ordered edit records, replacement bytes,
fixed output, and applied/skipped counts byte-for-byte across 66 implementation files
and 15 fixture inputs. Repeat the comparison against the concurrently updated rule.
Fixture inputs and expected output are not rewritten.

### Integration and reprofile

The rule changed concurrently during measurement. Rebase only the five-line guard
onto rule snapshot `04924638f5abd06b`; the integrated file matches the freshly tested
candidate byte-for-byte (SHA-256 starts `b6c4a652f40951f0`). The updated rule reports
4,488 diagnostics on the same frozen corpus, with 4,099 fixes; both arms agree.

A separate two-block ABBA/BAAB integration screen, using the same 16-batch/four-corpus
contract, confirms −4.05% rule instructions and −0.98% complete-lint instructions.
Complete-lint cycle change is −0.68%; wall time is +0.46%. This small screen is not
an independent latency verdict. Exact-output comparisons pass for all 81 inputs.

The original candidate's prepared-window reprofile has no lost samples and still
attributes approximately 35% to parsing and 21% to top-down declarations. There is
no evidence that this deletion turns generic AST dispatch into the main bottleneck.

Integrated checks pass: fixture/root tests in Debug and ReleaseSafe; ReleaseSafe
install and replay builds; formatting checks across linter and native sources.

## Experiment 2: cache function-body token boundaries

Hypothesis: compute each function body's inclusive first/last tokens during existing
function discovery, then retain the same linear caller search using integer bounds.
No additional AST pass, SIMD, owner cursor, or spelling cache. Replace the optional
body node rather than retaining both representations; prototype-only declarations
continue to have no body range.

The prior profile attributes roughly 5% self cycles to `Ast.lastToken` across all
callers, not just this rule; `firstToken` contributes roughly 2%. Eliminating all those
calls would bound speedup near 1.08×. This change removes only repeated body lookups
and adds eager one-time lookups, so its actual headroom is smaller.

Plan: fresh snapshots of the current implementation, exact diagnostics/edit/fixed
output comparison, two balanced four-process screening blocks, then eight ABBA/BAAB
confirmation blocks if promising. Use the same frozen 66-file input and fixed-work
zigbench contract as experiment 1. Reject if extra setup/storage outweighs the hoist.

### Confirmation

Same eight-block confirmation and analysis contract as experiment 1; all hardware
counter coverage readings are 100%. The baseline includes the empty-container guard.

| Boundary | Instructions | Cycles (interval) | Wall time (interval) |
| --- | ---: | ---: | ---: |
| Top-down rule | −7.90% | −6.40% (−8.87%, −3.13%) | −5.52% (−8.48%, −1.80%) |
| Complete lint | −1.86% | −1.84% (−4.52%, +0.81%) | −2.15% (−5.38%, +1.06%) |
| Complete lint + fixes | −1.79% | −3.24% (−4.60%, −2.05%) | −3.44% (−5.12%, −1.87%) |
| Directory lint | −1.83% | −3.65% (−6.11%, −1.50%) | −3.40% (−6.50%, −0.88%) |

Approximately 4.406 million instructions disappear per complete corpus. Branch misses
also decrease. The rule, fix-collecting workload, and directory workload improve in
this run; the plain preloaded-lint latency interval still crosses zero. Host contention
remains visible, so do not promise a universal percentage improvement.

Decision: retain. Net production change: +9 lines. Native `FunctionDeclaration`
remains 64 bytes, aligned to 8; cached bounds replace the optional node rather than
adding another array or allocation. Caller search order, inclusive containment, and
prototype-only behavior are unchanged. Existing comments remain intact.

Exact-output validation passes on 81 source/fixture inputs: diagnostics, ordered edits,
replacement bytes, fixed output, and applied/skipped counts all match. The production
source matched the measured baseline before integration; only the measured patch was
applied. Fixture files were not changed.

Integrated Debug/ReleaseSafe fixture tests, ReleaseSafe install/replay builds, and
formatting checks pass. The integrated rule matches the measured candidate byte-for-byte.

Fresh prepared-window profiles of both variants lose no samples. `Ast.lastToken`
self share drops from 4.85% to 3.64%; `firstToken` does not show a corresponding decline
(1.68% to 1.89%). Other rules still call these helpers, and eager boundary construction
is included. Treat these sampled shares as attribution, not a second speedup estimate.
The compiler also inlines the simplified function-order path in the candidate; no
manual inlining or layout directives were added.

## Experiments 3–4: caller cursor and known-identifier spelling

Test independent candidates against the current body-range-cached baseline, then
measure retained changes together with the separate token-line-index experiment.
The input remains the same frozen 66-file implementation corpus. Each screen uses
two ABBA/BAAB blocks; confirmation uses eight blocks. Each fresh process has three
warmups and eight measured batches of four corpora (fewer batches than experiment 2,
with the same number of independent confirmation processes). Predeclare each schedule;
retain raw counters, exact outputs, and host-condition records. No safety checks disabled.

### Caller cursor

Hypothesis: function bodies and lookup tokens are both source-ordered. A monotonic
cursor can discard bodies ending before the current token without rescanning previous
functions. Skip bodyless prototypes; keep the cursor when a token precedes the next
body. Recursing into a container starts its own cursor. No per-token owner map needed.

The previous function-order region consumes approximately 14% inclusive cycles;
removing the entire region would bound speedup near 1.16×. The search is only part of
that region, so this experiment's actual ceiling is lower.

### Identifier spelling

Hypothesis: the top-down rule's known `.identifier` tokens do not need the standard
tokenizer's dispatch and keyword lookup a second time. Scan the exact ASCII identifier
continuation alphabet to find the end. Quoted identifiers still use `Ast.tokenSlice`,
preserving escapes and source spelling. Assert the token-tag precondition; keep builtin
and other token handling unchanged. No new cache or prepass.

The earlier whole-linter `tokenSlice` inclusive share was about 12%; this targets only
top-down identifier calls, not all users. That is headroom, not a predicted speedup.

### Independent confirmation

Eight balanced blocks each; all PMU coverage readings 100%. Geometric within-block
ratios and exploratory 95% block-bootstrap intervals use the same analysis as above.

| Candidate / boundary | Instructions | Cycles (interval) | Wall time (interval) |
| --- | ---: | ---: | ---: |
| Cursor / rule | −0.86% | −1.42% (−2.87%, −0.53%) | −2.24% (−4.45%, +0.29%) |
| Cursor / complete lint | −0.19% | +0.45% (−1.34%, +2.76%) | +0.40% (−2.35%, +3.40%) |
| Cursor / lint + fixes | −0.18% | −1.36% (−2.70%, −0.47%) | −0.43% (−2.81%, +2.02%) |
| Cursor / directory | −0.19% | −2.35% (−6.89%, +0.91%) | −2.72% (−8.05%, +0.75%) |
| Identifier / rule | −27.99% | −13.33% (−14.82%, −11.14%) | −13.41% (−15.89%, −10.52%) |
| Identifier / complete lint | −6.20% | −1.91% (−3.23%, +0.06%) | −2.69% (−4.73%, −0.04%) |
| Identifier / lint + fixes | −5.96% | −2.61% (−4.31%, −1.28%) | −3.35% (−6.76%, +0.34%) |
| Identifier / directory | −6.10% | −2.93% (−5.33%, −0.96%) | −3.91% (−6.89%, −1.10%) |

The cursor deletes only 0.443 million instructions per corpus. Most of the inclusive
function-order cost was not the owner search. Its target-rule and fix-collecting cycle
reductions support keeping the small linear scan, but whole-directory latency remains
inconclusive in isolation. Do not describe it as a large speedup.

Known-identifier spelling deletes 14.380 million instructions per corpus. Retain the
narrow operation rather than adding an identifier cache. Other rules and quoted names
still use standard token slicing; this is not a general tokenizer replacement.

All independent candidates match exact diagnostics, edits, replacement bytes, fixed
output and applied/skipped counts on 94 cases: 81 source/fixture inputs plus 13 temporary
edge inputs. These cover prototypes, references outside bodies and in signatures, nested
containers, escaped/quoted/Unicode identifiers, BOM, CRLF, missing final newline,
multiline strings, and malformed input. A focused spelling-equivalence test inside
the existing rule file also passes; no new fixture files or snapshot rewrites.

### Combined gate and integration

Retain the cursor and identifier operation together with the line-run token index.
Eight balanced blocks with the same fixed-work contract; all PMU coverage 100%.

| Boundary | Instructions | Cycles (interval) | Wall time (interval) |
| --- | ---: | ---: | ---: |
| Prepared paragraphs | −9.97% | −26.52% (−28.70%, −24.78%) | −24.06% (−26.62%, −22.00%) |
| Top-down rule | −28.85% | −14.69% (−15.30%, −14.28%) | −16.09% (−17.82%, −14.32%) |
| Complete lint | −6.27% | −6.52% (−7.73%, −5.58%) | −7.87% (−9.94%, −5.97%) |
| Complete lint + fixes | −6.46% | −7.29% (−8.17%, −6.62%) | −6.10% (−7.42%, −4.92%) |
| Directory lint | −6.17% | −6.90% (−8.43%, −5.60%) | −7.06% (−9.40%, −4.79%) |

Raw process medians for complete lint are 25.88 → 24.22 ms; directory medians are
27.44 → 25.52 ms. These descriptive medians are not the paired-block estimators above.
Complete-lint branch misses fall approximately 12.46%. Costs include allocating,
building, querying, and releasing the token-line table. No manual code-layout or
runtime-safety changes were made.

Across the four implementation files, net change is +59 lines, including the 27-line
spelling test (+32 non-test lines). Existing comments are preserved. Integrated files
match the final measured candidate byte-for-byte. Exact comparisons pass on all 94
inputs; complete linter tests pass in Debug and ReleaseSafe, including the new spelling
test and existing scalar/AST-location checks. Native ReleaseSafe install/replay builds
and repository formatting checks pass.

The final prepared-window A/B profiles lose no samples. Standard tokenizer self share
falls from 27.06% to 20.13%; standard `tokenSlice` inclusive share falls from about
11.35% to 3.42%. The new narrow identifier scan itself now accounts for 6.49% self
samples. Parsing remains about 37.53% inclusive, and memory initialization is 12.16%
self in the candidate. These are overlapping attribution figures, not additive costs
or independent timing evidence; the next investigation should use the new profile.

## Remaining targets

- Reprofile remaining standard token-slice callers before extending the narrow
  identifier operation beyond this rule. Do not infer payoff from call-site count.
- The cached body ranges and monotonic cursor leave little owner-search work;
  prefer another measured region over a more elaborate owner index.
- Do not infer a shared AST visitor win from the number of loops. The optional-unwrap
  rule already scans the full AST in roughly 0.1 ms; shared dispatch cannot remove
  the parsing, token spelling, graph, and reporting work dominating this corpus.

## Current analysis: avoid negative function-name lookups

Status: investigation only. No production rule changes or new test files.
The rule exactly matches the latest complete-lint profile snapshot. That capture
uses Zig 0.16.0 native ReleaseSafe, CPU 8 on the i7-13700HX P-core (sibling 9),
600 complete corpora after ten warmups, acknowledged perf windows, and 14,342
samples with none lost. The complete boundary and host limitations are recorded
in `../Linter.perf-log.md` under the current shared-tail attribution. The isolated
prepared rule costs about 16.05 M cycles / 4.76 ms per 66-file corpus and emits
13 diagnostics. This investigation does not introduce an A/B performance claim.

### Which part of the rule is hot?

Period-weighted complete-lint stacks assign 19.49% of total cycles to this rule.
Partitioning those samples by rule phase gives:

| Phase | Whole-lint cycles | Rule cycles |
| --- | ---: | ---: |
| Function references and caller analysis | 13.29% | 68.22% |
| Container setup, tables, and remaining work | 2.95% | 15.15% |
| Signature analysis | 2.59% | 13.30% |
| Type ownership and contract locality | 0.65% | 3.33% |

Identifier spelling alone accounts for 7.12% of whole-lint cycles beneath this
rule, or 36.56% of its cost. Caller-attributed string-map probing and Wyhash
execution account for about 3.10 and 1.77 additional percentage points. These
families dominate; the final ordering comparisons and monotonic caller cursor do
not. Parameter-name walking is 1.15% of the whole lint, and the rule's memory fills
are 0.70%. These cross-cutting figures are included in the phase table, not added
to it. The misleading LintCache-typed lookup symbol is shared generic machine
code called by this rule's name maps, not activity in the disabled disk cache.

### Untimed work census

Temporary instrumentation runs the original rule without skipping or changing any
lookup, reference classification, or report. Counters reset after preparation;
the measured census is one pass over the same frozen 66-file corpus.

- 348 containers, 3,321 members, 880 named declarations, 292 types, 805 functions.
- The existing zero-function guard leaves 122 function-reference scans.
- Those scans visit 221,035 tokens and spell/hash 70,005 identifiers, covering
  558,666 identifier-source bytes including repeat visits.
- Only 2,328 lookups hit: **96.67% miss**. Of the hits, 805 are the function's own
  declaration, 247 are outside any recognized caller body, 481 are non-call or
  qualified references inside a body, and 10 are self-calls.
- 61,085 distinct identifier tokens are visited; 8,920 visits repeat a token in
  another container scan (12.74%). Repeated nesting is real but does not explain
  most lookup work, so a file-wide spelling cache is not the first intervention.
- 941 hits already target an ambiguous function. An early exit after such a hit
  would only avoid cheap post-lookup work; it cannot address the dominant misses.

The three declaration arrays reserve 425,088 payload bytes across all containers
and use 89,024 bytes for populated records. These are cumulative requested payload
sizes, not peak retained memory, allocator capacity, or complete rule allocation.
Despite the approximately 79% unused slots, memory fills own only 0.70% of total
cycles here. Do not lead with a multi-pass sizing or arena redesign to optimize
this smaller region while name lookups dominate.

### First candidate: prove a scan cannot report

A function-order diagnostic requires a private function before a later function
whose body contains the sole caller. If no private function precedes any later
function body, this pass cannot report. A reverse walk over existing function
metadata can establish that without reading source tokens or allocating storage.
The private target need not itself have a body; only the possible caller does.

Counted without changing behavior:

- Fewer than two functions: 37 scans / 6,377 identifier visits.
- All functions externally visible: 50 scans / 9,801 identifier visits.
- General no-possible-reversed-caller condition: **71 scans / 15,177 visits**,
  eliminating 21.68% of identifier lookups. The narrower categories overlap and
  are included in this total; they must not be added together.

Put this guard inside `lintFunctionOrder`, not around the container pass. Signature
and type checks, plus recursion into nested declared containers, must still run.
Caller state is consumed only by the same function-order pass, so skipping this
provably non-reporting analysis does not deprive another phase of data.

### Second candidate: cheap negative prefix filtering

Before spelling an identifier or probing the function-name map, test whether its
first source byte(s) could belong to any function name in this container. Build
this filter from the names already collected; no whole-file prepass is needed.
Possible membership still runs the existing exact spelling/map lookup and all
reference classification. False positives are harmless; false negatives are not.

The census evaluates three filters while still performing every real lookup:

| Negative filter | Identifier visits rejected |
| --- | ---: |
| Exact first-byte membership | 41,133 / 70,005 (58.76%) |
| Exact first-two-byte membership | 56,636 / 70,005 (80.90%) |
| 256-bit first-two-byte hash filter | 55,583 / 70,005 (79.40%) |

The fixed illustrative hash is `(first_byte * 33 ^ second_byte) & 255`; it was not
tuned to the corpus. Instrumentation uses Boolean arrays to count membership;
a production 256-bit filter would occupy 32 bytes, with an additional conservative
fallback for one-byte names. Exact two-byte membership is only a selectivity
reference, not a recommendation for a large per-container table.

After the impossible-order guard, the 256-bit filter rejects another 41,826 of
54,828 remaining visits (76.28%). Together they avoid 57,003 of the original
70,005 spelling/map probes (81.43%). Both pair filters reject zero actual hits in
the census. This establishes selectivity, not throughput or a complete correctness
gate; compare byte-for-byte diagnostics and fixes on fixtures before retention.

Important constraints:

- Do not filter to call syntax alone. Matching non-call references and references
  outside a caller body deliberately mark ordering ambiguous; omitting them could
  introduce diagnostics. The 247 outside-body and 481 non-call hits demonstrate
  this is not merely theoretical.
- Match existing raw source spelling, including quoted and escaped identifiers.
  Quoted names sharing the `@\"` prefix may pass conservatively; do not decode or
  normalize them as part of this optimization.
- One-character function names must never be rejected based on the following
  punctuation or whitespace byte. Reads at physical EOF must remain bounded.
- Keep tracking externally visible functions when a private diagnostic is still
  possible: their caller/ambiguity state determines whether they qualify as roots.
- Skipping failed prefix matches need not advance the caller cursor; the next real
  match already advances it monotonically to that token's source position.

### Smaller follow-up: reject unknown signature names earlier

`lintSignatureIdentifier` currently performs parameter-shadow checks before the
local declaration lookup. Of 2,896 signature identifiers, 1,834 do not name a local
declaration. Not all are expensive: primitives already short-circuit many checks.
The precise census finds 392 of 1,454 parameter walks, visiting 883 of 4,366
parameters, occur for names absent from `named_by_name`.

The type-name map is a subset of the named-declaration map, so an early named-map
miss can safely avoid those shadow walks and the later type lookup. Keep all
primitive/Self/shadowing exclusions before updating consumer facts or reporting.
This is a smaller candidate than function-reference filtering; it does not justify
a new per-function parameter-name cache before measuring the simpler gate.

### Proposed experiment order

1. Benchmark the impossible-order guard alone against the current snapshot.
2. Compare first-byte and small two-byte negative filters on the remaining scans;
   include setup and the complete lint boundary, not just saved query counts.
3. Try the early signature-name miss gate separately if the larger changes land.

The function-reference phase owns about 13.3% of total cycles, which bounds its
complete deletion's headroom. An 81% reduction in probe count is **not** an 81%
rule or linter speedup: filter setup, source/tag scanning, successful lookups, and
other phases remain. Preserve the established diagnostics/ambiguity semantics and
use balanced zigbench A/B measurements before making any performance claim.

## Experiment 5: impossible-order guard, prefix filter, and signature gate

Status: **retained**. The investigation above preceded this implementation.
Baseline rule SHA-256:
`1c8a47ca368eef4f0dad4a984f70eb6c9856f275df761ada8b2875f659ec9cf7`.
All arms freeze the same current linter implementation; only this rule differs.
The frozen input remains 66 files / 1,114,088 bytes / 97,485 nodes / 197,393 tokens.
The current snapshot reports 4,490 diagnostics and 3,995 proposed fixes; the
prepared top-down rule emits 13 diagnostics. Historical counts above belong to
older rule snapshots, not discrepancies between these arms.

### Changes and correctness boundary

1. Reverse-walk the existing function metadata. If no private function precedes
   a later body, return from function-order analysis only. A bodyless private
   prototype can still be the target; signatures, types, and recursion still run.
2. Build two `StaticBitSet(256)` values from collected names: the hashed two-byte
   prefix filter and conservative one-byte-name membership. Together they use
   64 bytes of local payload, with no heap allocation or persistent index.
   Reject impossible prefixes before identifier spelling and exact map lookup.
   Preserve raw quoted/escaped spelling, bounded second-byte access, and every
   existing exact-match/reference-ambiguity check, including public root callers.
3. Look up signature names in `named_by_name` before reconstructing parameter
   shadowing. Its miss also proves a type-map miss. All primitive, `Self`, and
   shadowing exclusions still precede consumer updates and diagnostics.

The non-test patch adds 77 lines and removes four, net +73. Existing comments
remain intact. No parser/tokenizer changes, parallelism, safety relaxation,
additional declaration columns, or global reference cache.

All four candidates (each change alone and the combination) match the baseline
on **1,436 inputs in Debug and ReleaseSafe**: indexes, diagnostics, ordered edits,
replacement bytes, fixed output, and applied/skipped counts. These include 1,324
new temporary generated cases. The collision generator initially emitted `fn()`;
validation rejected that malformed source rather than counting equal parse errors
as coverage. Adding `Target`/`Other` suffixes made all 256 collision cases legal
without changing their first-two-byte collisions. Original failure logs remain.

A 23-case regression test was added inside the existing rule file. It pins private
and public order, bodyless prototypes, single-byte names, hash collisions, quoted
names and raw escape mismatch, non-call/qualified references, ambiguous public
callers, nested containers, unknown signature names, parameter shadowing, and
signature checks despite an impossible function-order scan. No fixture snapshots
or new production test files were added. Before that test was appended, the
integrated rule matched the measured combined candidate exactly.

### Isolated screens

Each candidate gets two balanced ABBA/BAAB blocks, three warmups and eight measured
batches of four complete corpora per process. Zig 0.16.0 native ReleaseSafe,
i7-13700HX CPU 8 P-core, sibling 9, single-threaded; no lint-result cache.

| Candidate | Instructions removed / corpus | Complete-lint instructions | Rule cycles |
| --- | ---: | ---: | ---: |
| Impossible-order guard | 5.067 M | −2.379% | −20.147% |
| Two-byte prefix filter | 14.933 M | −7.012% | −38.907% |
| Early signature lookup | 0.417 M | −0.196% | −1.460% |

These are screens, not precise standalone latency estimates. The guard's first
block was unusually noisy; its large paired complete-lint cycle estimate is not
representative of the raw medians (81.618 M → 78.565 M). The prefix screen's
complete-lint cycle reduction was −7.128% in two consistent blocks. Signature
lookup removes work and improves isolated rule cycles, but its complete-lint
cycle change (+0.164%) is indistinguishable from noise. Do not claim a separate
whole-lint speedup for the signature gate. The three changes overlap, so their
instruction savings must not be summed.

### Combined confirmation

Eight predeclared balanced four-process blocks, with the same warmup/batch/work
contract. Percentages are geometric within-block B/A ratios; intervals are
exploratory 95% bootstraps over the eight blocks, not independent batch samples.
All hardware-counter coverage readings are 100%.

| Boundary | Instructions | Cycles (interval) | Wall time (interval) |
| --- | ---: | ---: | ---: |
| Prepared top-down rule | −46.674% | −45.335% (−49.614%, −42.460%) | −44.947% (−50.532%, −40.958%) |
| Complete lint | −7.927% | −8.290% (−10.618%, −5.953%) | −7.905% (−10.757%, −5.111%) |
| Complete lint + fixes | −7.571% | −9.126% (−13.530%, −5.160%) | −9.029% (−14.163%, −4.093%) |
| Directory lint | −7.797% | −8.885% (−11.713%, −5.729%) | −8.483% (−11.967%, −4.310%) |

Raw complete-lint process medians: **80.825 M → 73.751 M cycles**,
**24.051 → 22.028 ms**, and **212.971 M → 196.089 M instructions**.
The combination removes 16.882 M instructions per corpus. These raw medians are
not the paired-block estimators in the table. The isolated rule excludes its
prerequisites; complete lint includes parsing, context construction, all rules,
diagnostics, allocation, and teardown. Directory lint also includes warm OS
reads. None of these is a cold-disk or startup measurement.

Host caveat: 127 of 128 stage observations report thermal-throttle increases,
125 report sibling activity, and effective frequency spans 3.07–3.79 GHz. No
frequency-policy changes occurred. All observations and unfavorable blocks are
retained. Instruction reduction, consistently favorable complete-lint blocks,
and the bounded combined intervals support retention on this workload; they do
not establish a universal desktop latency percentage.

### Generated code and reprofile

The native disassembly confirms a reverse 64-byte-record walk before filter setup,
two 32-byte zero stores, inline shift/add/XOR prefix hashing, and bit tests whose
negative branches bypass `identifierText` and string-map probing. Second-byte
bounds/overflow checks remain. The signature map miss branches out before either
parameter-name walk. Generic map code is again labeled `LintCache.PathMetadata`
in places because the compiler shares implementations; the disk cache is off.

The bitset value methods still emit a 32-byte stack copy at each membership query.
The measured win includes that cost; this is not an idealized single-load filter
or a claim that the generated loop is fully optimized. No follow-up representation
change was bundled into this experiment without its own measurements.

Separate before/after profiles cover 600 complete no-fix corpora after ten warmups,
with acknowledged perf windows. The captures contain 14,841 / 13,084 samples and
zero lost samples. Two baseline and one candidate samples lack callchains; they
remain in the unattributed denominator. Tail-call ownership uses each binary's
own disassembled Linter return addresses, not addresses from an older build.

| Disjoint owner | Before | After |
| --- | ---: | ---: |
| AST construction | 37.72% | 42.59% |
| Top-down declarations | 20.41% | 12.20% |
| Visible control flow | 16.22% | 17.59% |
| Statement paragraphs | 15.80% | 16.85% |

Overlapping identifier-spelling share falls from 9.56% to 4.53%; memory fills are
13.07% → 14.31%. These shares are attribution, not another latency estimate, and
growing relative shares do not demonstrate regressions in untouched components.
Parsing is still the largest owner; generic rule dispatch has not become the
principal bottleneck.

Integrated checks pass: focused regression tests and full project tests in Debug
and ReleaseSafe, rule/native formatting, and the ReleaseSafe install build.
The older documented `install replay` command failed because concurrent build
changes removed the `replay` step; the supported install was rerun successfully.
The removed step was not restored, and unrelated work was not overwritten.

Evidence directory:
`/tmp/experiments/voiced-lint/baseline-20260908/evidence/20260909T015511Z-top-down-filters`.
It retains patches, source/binary hashes, generated inputs, exact-output validators,
raw zigbench schedules/invocations/JSON, block summaries, host conditions, perf
captures/stacks/reports, disassembly, analysis scripts, and check records.
