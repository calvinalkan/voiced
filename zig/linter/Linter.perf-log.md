# Cold-lint storage investigation

## Scope and measurement contract

Investigation only; no production storage or allocator API change was integrated.
Zig's parser, tokenizer, allocator wrappers, and safety settings were not modified.

- Frozen 66-file corpus: 1,114,088 source bytes, 97,485 AST nodes, 197,393 tokens.
- Frozen implementation produced 4,494 diagnostics and 4,099 fix proposals.
- Zig 0.16.0, native ReleaseSafe, Intel i7-13700HX P-core logical CPU 8.
- Lint cache disabled. Complete parsing, context construction, rules, reporting,
  and file-scratch teardown included. Sources preloaded for `lint_file`; directory
  scans include traversal and reading with warm OS caches.
- “Fresh storage” means construction inside each measured scan, not cold disk access
  or process startup. Prepared benchmark corpus construction is outside timing.
- Concurrent rule/fixture edits after the snapshot were preserved, not incorporated
  into one side of a comparison. Results describe the frozen implementation.

## Attribution before changing anything

Acknowledged perf windows cover 300 complete corpora after ten warmup corpora,
excluding preparation and verification. No lost samples. Shares below are approximate
percentages of total sampled CPU, not percentages of the `memset` symbol alone.

| Work | Share |
| --- | ---: |
| AST construction, inclusive | 37.4% |
| `compiler_rt.memset`, self | 11.4% |
| Memset under AST construction | 2.9% |
| Memset under AST destruction | 2.4% |
| Memset under diagnostic-array growth | about 2.3% |
| Memset under FileContext allocation/destruction | about 1.1% |

The remaining memset callers include rule buffers, hash-table storage, and logical
initialization. These rows overlap the inclusive AST row; do not sum them all.

`Allocator.alloc` and `free` contain undefined-memory poisoning. The profile confirms
real writes at those call sites in this ReleaseSafe build. An arena or fixed-buffer
backend does not bypass the wrappers. Arena retain-capacity reset itself does not
clear its retained backing storage. Required flag resets are a separate cost.

Consequently, even eliminating every memset would save at most the observed 11.4%
of CPU time. A diagnostic-storage change targets a much smaller share, plus copying
and growth overhead. It cannot remove the AST's internal allocation writes.

## Hidden AST helper work

The installed standard-library implementation has these relevant behaviors:

- `tokenSlice` retokenizes variable-spelling tokens; fixed lexemes return directly.
  Remaining tokenizer self time under the assertion-header rule is about 2.1% of
  the full workload, and under paragraph checks about 1.1%.
- `firstToken` and `lastToken` follow node boundary chains rather than reading
  cached boundaries. Their self shares are about 1.8% and 4.2%. They do not generally
  walk every node in the subtree.
- `fullIf`, `fullWhile`, and `fullFor` use `lastToken` to recover payload and else
  tokens. Repeated normalization can repeat those walks. Presence-only callers are
  candidates for tag checks; generated code still needs checking because unused
  reconstruction can sometimes be optimized away.
- `fullVarDecl` and `fullFnProto` scan adjacent declaration modifiers. These are
  short local scans, not source-file rescans or heap allocations.
- Function-parameter iteration recovers names, modifiers, docs, and boundaries.
  Signature traversal currently discards everything except `type_expr`; iterating
  `proto.ast.params` directly is a narrower candidate for those particular loops.
  Name-aware consumers still need parameter-name handling.
- Standard AST location/same-line helpers scan source bytes. Our FileContext index
  already replaces those operations in normal diagnostic and spacing queries.

No new helper cache or shared AST dispatcher was implemented in this investigation.

## Diagnostic storage hypothesis

Source-line copies occur between diagnostic appends in the same arena. This prevents
ordinary in-place array growth. An untimed allocation census confirmed:

| Quantity | Bytes/count |
| --- | ---: |
| Diagnostic size | 120 bytes |
| Array growth events | 18 |
| Growth events moving the buffer | 18 |
| Total allocated array capacity | 2,237,280 bytes |
| Discarded old array capacity | 1,488,360 bytes |
| Copied diagnostic source text | 298,609 bytes |

The experiment supplies 8,192 typed slots through the existing ArrayList buffer API:
983,040 bytes, or 960 KiB. This is a corpus-specific experimental bound, not a default
or a production hard-limit implementation. Preparation verifies the complete result
fits, and each scan asserts the list retained its original pointer and capacity.

Three modes share one benchmark binary:

1. Growing: unchanged diagnostic-array growth.
2. Reserved: allocate the typed buffer at scan start, inside timing; release its
   arena inside timing too. Reuse its slots across the scan's 66 files.
3. Reused: allocate once during workload preparation; each scan starts a new empty
   list over those slots. Initial allocation and final release are outside timing.

All modes still copy and retain source text. Scratch for parsing and rules is unchanged.
The difference between reserved and reused must not be presented as a cold-scan win.

## Balanced confirmation

Each comparison uses eight predeclared ABBA/BAAB four-process blocks: three warmup
batches, eight measured batches, four full scans per batch. Summaries use geometric
within-block ratios; exploratory 95% bootstrap intervals resample blocks, not batches.
PMU coverage was 100%. No frequency-policy changes occurred, but thermal events and
SMT-sibling activity were present. The reservation cohort had especially substantial
sibling contention; no observations were dropped.

| Storage / boundary | Instructions | Cycles (95% interval) | Wall time (95% interval) |
| --- | ---: | ---: | ---: |
| Reserved / complete lint | −1.79% | +0.97% (−1.67%, +5.22%) | +0.83% (−3.73%, +7.44%) |
| Reserved / lint + fixes | −1.73% | −0.05% (−3.65%, +5.55%) | −3.59% (−7.70%, +2.72%) |
| Reserved / directory | −1.76% | −1.39% (−5.31%, +3.04%) | −1.35% (−5.96%, +4.09%) |
| Reused / complete lint | −2.39% | −3.21% (−4.20%, −2.13%) | −2.33% (−4.47%, +0.09%) |
| Reused / lint + fixes | −2.30% | −3.18% (−3.93%, −2.38%) | −5.04% (−6.22%, −3.83%) |
| Reused / directory | −2.35% | −3.38% (−4.12%, −2.58%) | −4.50% (−6.77%, −2.01%) |

Reservation demonstrably deletes work, but this experiment does not establish a
first-scan latency improvement. Cross-scan reuse has a clearer modest cycle reduction.
It is not evidence for a large cold-time improvement from a general workspace rewrite.

A follow-up profile of reservation removes the diagnostic-array growth stacks from
significant memset attribution; the single initial buffer allocation accounts for
about 0.6% instead. Total memset self share is 10.1%, versus 11.4% before. Profile
shares are attribution evidence, not the acceptance estimate.

## Correctness and next decision

Debug and ReleaseSafe exact-output comparisons pass for 95 frozen inputs: corpus,
fixture inputs, and additional malformed/quoted/nested/CRLF cases. Comparisons include
rendered diagnostics, ordered edit bytes, replacements, fixed output, and fix counts.
The candidate diagnostic buffer is reused across validation inputs, including empty
and malformed sources. Performance-run counts also match in every observation.

No production code was changed. Next cold-time candidates should remove unnecessary
AST helper work before adding a general scratch ownership framework. A bounded
workspace remains possible as a separate resource-policy feature:

- A fixed arena backing buffer bounds allocator consumption, not poisoning cost.
- Retained typed rule arrays avoid repeated alloc/free wrappers, but logical state
  still needs resetting and recursive container lifetimes must not overlap.
- Source size is not a diagnostic-count or retained-text budget. Exhausting a hard
  budget must return an explicit failure, never truncate diagnostics into a clean
  result or publish a clean cache record.
- The observed largest control-rule scratch footprint without fixes was only
  78,210 bytes; the largest source was 105,498 bytes. Neither is a worst-case bound
  for arbitrary input at the configurable file-size limit.

## Follow-up: remove unnecessary AST helper work

This follow-up integrates two local work reductions, not the storage experiment.
A fresh implementation snapshot uses the same frozen 66-file input corpus. It emits
4,491 diagnostics and 4,099 fixes; concurrent rule changes explain the different
count from the earlier investigation. Both arms use this same fresh implementation.

### Isolated mechanisms

Each candidate first received two balanced ABBA/BAAB four-process screening blocks,
with the same native ReleaseSafe build and measurement boundaries as above. These
short screens establish instruction changes, not precise latency estimates.

| Candidate | Target-rule instructions | Complete-lint instructions | Decision |
| --- | ---: | ---: | --- |
| Replace six presence-only `fullIf` calls with a tag helper | +1.43% control | +0.30% | Revert |
| Iterate signature type nodes directly | −3.20% top-down | −0.55% | Retain |
| Share known-identifier spelling | −36.99% assertion-header; −3.97% paragraphs | −1.72% | Retain |

The tag helper did remove six static `fullIf` call sites, and the helper itself was
inlined. Nevertheless, isolated dynamic instruction counts increased by about
657,000 per corpus and complete-lint cycle differences were inconclusive. Inspection
also found other generated-code differences, including an additional static memset
call site in the profiling build; this does not by itself attribute the dynamic
increase. Do not assume removing an AST helper call guarantees faster machine code.
The original control rule was restored byte-for-byte rather than retaining an
unsupported optimization or adding inlining/layout directives.

The two retained parameter loops consume `proto.ast.params` directly. Parameter-name
checks still use the standard iterator. The known-identifier operation moved intact,
including its implementation comments, from the top-down rule into FileContext as an
AST-taking namespace function. This avoids threading line-index context through
AST-only helpers. Assertion-header and paragraph checks now share it; quoted names
still use the standard tokenizer and keep their original source spelling. Builtins,
strings, and token slicing for fix ranges are unchanged. No indexes or allocations
were added. The four edited implementation files have seven fewer lines in total.

### Combined confirmation

The retained pair received eight predeclared ABBA/BAAB four-process blocks. Each fresh
process uses three warmup batches and eight measured batches of four corpora each.
The baseline and candidate are independent binaries. Rows below are geometric paired
block ratios, with exploratory 95% bootstrap intervals over whole blocks.

| Boundary | Instructions | Cycles (95% interval) | Wall time (95% interval) |
| --- | ---: | ---: | ---: |
| Prepared rules | −4.08% | −2.69% (−4.19%, −0.73%) | −2.91% (−5.09%, −1.13%) |
| Complete lint | −2.27% | −2.38% (−3.02%, −1.94%) | −2.11% (−3.37%, −0.84%) |
| Complete lint + fixes | −2.19% | −2.28% (−2.83%, −1.86%) | −2.85% (−4.48%, −1.29%) |
| Directory lint | −2.23% | −0.50% (−2.87%, +3.70%) | +0.49% (−3.64%, +7.02%) |

PMU coverage was 100%, with no frequency-policy changes. Thermal events and SMT-sibling
activity remained present; one directory block was notably noisy. No observations
were removed. Complete-lint improvement is supported; directory throughput/latency
improvement is not established by this cohort despite its reduced instruction count.
The result includes parsing and scratch construction/destruction, not just prepared
rule execution. It does not measure cold disk access or process startup.

### Reprofile, validation, and integration

Acknowledged profiling windows again cover 300 complete corpora after warmup, with
no lost samples. Tokenizer self share falls from 20.16% to 16.43%. The shared identifier
operation rises to 8.95% self share, now including the assertion-header and paragraph
callers as well as top-down checks. This confirms that spelling work moved to the
narrow operation; it was not eliminated entirely. Profile shares are attribution,
not the performance acceptance estimate.

Exact comparisons pass in Debug and ReleaseSafe on 103 inputs: the frozen corpus,
current fixture inputs, and temporary edge cases. They compare rendered diagnostics,
ordered edits, replacement bytes, fixed source, and applied/skipped counts. Added
edges cover interleaved `anytype`, varargs, shadowing type parameters, nested function
types, parameter docs/modifiers, capture/else-if chains, and quoted assertion aliases.
The eight added valid-syntax cases are also explicitly checked for parse errors.
No fixture snapshots or production test files were changed.

Root tests pass in Debug and ReleaseSafe, as do formatting and native ReleaseSafe
install/replay checks. All four edited implementation files match the final measured
candidate; the control rule matches the frozen baseline. Existing identifier tests
exercise the relocated helper through its imported alias. No existing implementation
comment was discarded, and unrelated worktree edits were preserved.

## Current cycle attribution after the shared SIMD tail

Status: investigation only; no production code changes or new test files.
This refresh uses a new current linter snapshot, including the two-region SIMD
FileContext constructor. The frozen corpus remains 66 files / 1,114,088 bytes,
97,485 AST nodes, and 197,393 tokens. The current surrounding rules produce 4,490
diagnostics and expose 3,995 fixes during untimed preparation; fix collection is
**disabled** in the measured complete-lint work. Do not compare those fix totals
with older snapshots as though the surrounding implementation were unchanged.

### Capture and counter contract

Native Zig 0.16.0 ReleaseSafe, single-threaded on Intel i7-13700HX P-core logical
CPU 8, physical core 16, SMT sibling 9. A symbolized frame-pointer build records
cycles:u at 997 Hz around 600 complete corpora after preparation and ten warmup
corpora. Perf enable/disable handshakes are acknowledged. The window contains
parsing, context construction, rules, diagnostics, allocation, and per-corpus
teardown; it excludes file reads, the lint cache, fixture preparation, and final
persistent-corpus teardown. There are 14,342 samples, no lost samples, and a total
sample period of 49,059,652,393 cycles. This is attribution, not an A/B result.

A separate unprofiled zigbench capture uses three warmup batches and eight measured
batches of four corpus operations, including isolated diagnostic stages. Its full
lint median is **80.710 M cycles / 23.483 ms**, retiring 212.863 M instructions,
41.476 M branches, and 859,357 branch misses per corpus. Aggregate IPC is 2.637.
All counter coverage is 100%. One process's batches do not supply independent
comparison trials or a confidence interval.

The observed policy is intel_pstate / powersave, performance energy preference,
turbo enabled, and configured 0.8–5.0 GHz bounds. The full-lint counter capture
averages 3.418 GHz, observes no policy changes, and has thermal-counter increases
in all eight measured batches and some sibling activity in seven. Nothing is
filtered out. Wall-time precision and any profile-to-unprofiled extrapolation
remain limited by host conditions and profiling-build differences.

### Non-overlapping ownership of sampled cycles

Each sample is weighted by its recorded event period and assigned once to its
owning operation. Inline frames are retained. Where a tail-called helper omits
an owner frame, the return address in Linter identifies the direct call in this
exact binary's disassembly; this recovers about 2.35% of the total. The remaining
unattributed/lifecycle bucket is 0.13%. These ownership totals include callees,
allocation, and diagnostics inside each operation.

| Owner | Sampled cycles |
| --- | ---: |
| AST construction, including tokenizer and allocation | 38.98% |
| top_down_declarations | 19.49% |
| visible_control_flow | 16.10% |
| visible_statement_paragraphs | 15.12% |
| assert_header_snapshot | 2.97% |
| FileContext construction | 2.88% |
| AST teardown | 2.64% |
| visible_type_declarations | 0.82% |
| FileContext teardown | 0.65% |
| explicit_optional_unwrap | 0.24% |
| Other lifecycle / unattributed | 0.13% |

The three large rules account for 50.70% together. FileContext construction and
release together account for 3.53%; making its constructor free would remove at
most its observed 2.88% without changing work elsewhere. This is a sampled Amdahl
ceiling, not a prediction that such a change is achievable.

### Cross-cutting costs: already included above

- Tokenizer execution is 17.08% inclusive, of which 16.76 percentage points are
  beneath AST construction. It is not an additional 17% on top of parsing.
- Identifier spelling is 9.35% inclusive. Directly visible rule frames attribute
  7.05 points to top-down declarations, 1.50 to assertion headers, and 0.68 to
  paragraphs; another 0.12 points omit the immediate rule frame.
- First/last-token boundary helpers together are 5.73% inclusive. Native self
  rows assign 4.01% to lastToken and 1.71% to firstToken.
- Diagnostic operations are 5.00% inclusive across their calling rules.
- Memset-family samples total 12.97%, including 12.45% in the native memset
  implementation. These fills are already charged to their callers: AST
  construction 3.50 points, paragraphs 2.92, AST teardown 2.64, control flow 1.84,
  top-down declarations 0.70, context construction 0.67, context teardown 0.65,
  and the remaining operations about 0.06. This includes ReleaseSafe allocator
  fills and explicit initialization; it is not all one removable clearing pass.

Native self rows also show 3.10% in a string-hash-map lookup and 1.81% in Wyhash.
The lookup symbol happens to contain `LintCache.PathMetadata`, but its callers
are the top-down rule's `StringHashMap(usize)` operations. Equivalent generic
machine code shares a symbol here; this is **not** evidence of the disabled lint
cache running inside the window. Attribute it from the callers, not that name.

### Isolated stage counters and next target

| Prepared boundary | Median cycles per corpus | Median time |
| --- | ---: | ---: |
| Tokenization only | 13.911 M | 3.737 ms |
| AST parse + release | 32.202 M | 8.884 ms |
| FileContext build + release | 2.514 M | 0.744 ms |
| Prepared control-flow rule | 12.782 M | 3.812 ms |
| Prepared paragraph rule | 11.540 M | 3.608 ms |
| Prepared top-down rule | 16.050 M | 4.763 ms |
| All prepared rules | 45.000 M | 14.158 ms |
| Complete parse + lint | 80.710 M | 23.483 ms |

Prepared boundaries omit their prerequisites, retain different working sets, and
may compile/invoke operations differently. They corroborate the ranking, but must
not be added together to manufacture a full-workload breakdown. Profile shares
above come from the actual complete-lint stack samples instead.

The next controllable area is top-down declaration/name work, followed by the
control-flow and paragraph rules, not another small constructor adjustment.
Parser changes remain out of scope. Before implementing a new cache or prepass,
measure the relevant repeated name/boundary queries and their caller requirements;
the rejected empty-line index already demonstrates that hoisting unused work can
cost more than the queries it removes.

## Borrowed fix replacements

A 2026-09-10 A/B benchmark compared copied replacement bytes with edits that borrow
file-arena-backed replacements. The synthetic 205 KiB source contains 6,000 unbraced
`if` exits; each ReleaseFast process performs one public explicit-file lint with fix
collection and formatting.

Two 100-run hyperfine orders measured the copied baseline at 10.3/10.2 ms and the
borrowed candidate at 10.1/10.1 ms. This only establishes no visible wall-time
regression at that scale. A 50-run pinned PMU capture reported 126.606 M versus
125.558 M P-core instructions (-0.83%) and 21.139 M versus 20.972 M P-core branches
(-0.79%); hybrid-core scheduling made cycle measurements unsuitable for comparison.

The borrowed representation is retained. It removes the replacement-byte array,
its growth allocations, and one copy per edit. `StoredEdit` grows from 16 to at most
24 bytes, and replacement slices remain valid because production rules allocate them
from the file arena that owns the complete fix plan. The benchmark corpus and binaries
were throwaway artifacts outside the repository.
