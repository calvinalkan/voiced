# Visible control-flow performance

## Experiment 1: tag-directed context classification

Status: **retained**. The complete-lint benchmark remains the acceptance boundary;
prepared-rule measurements help locate costs but do not replace it.

### Baseline and hypothesis

The rule changed concurrently after the previous whole-linter profile. Freeze a
fresh common implementation snapshot for both arms rather than attribute those
changes to this experiment. Baseline rule SHA-256:
`4e83b13b1d3aaaa30afd97fea2e511cca799b6394173c4480590f166c23fd208`.

The fresh complete-lint profile attributes 13.73% of cycles to control flow,
including 6.31% to context classification (45.96% of the rule). Complete removal
would bound speedup at about 1.16× for the rule or 1.07× for classification;
this change removes only unnecessary decoder work within classification.

The first classification loop previously tried `blockStatements`, `fullIf`, and
`fullSwitchCase` for every node. Their supported tags are disjoint. Switch on the
tag already supplied by the node scan, then invoke only the applicable decoder.
Retain all block-child marking, `else if` handling, switch-arm marking, and the
subsequent worklist propagation unchanged. Keep the standard Zig AST decoders;
this is an optimization of their use, not a parser fork or shared rule dispatcher.

Untimed census on the fixed corpus:

- 97,485 nodes; 2,614 blocks, 1,654 conditionals, and 1,094 switch cases.
- First-pass decoder requests: **292,455 → 5,362** (98.17% fewer).
- These are source-level requests, not machine-call counts or a CPU speedup.
  Conditional `else if` queries and worklist decoding remain unchanged.
- The corpus has no multi-item inline switch case. Temporary generated inputs
  deliberately cover that tag and every other supported tag: all four block,
  two conditional, and four switch-case representations.

### Measurement contract

Zig 0.16.0 native ReleaseSafe; i7-13700HX CPU 8 P-core, sibling 9; single-threaded
linting with runtime safety enabled and no lint-result cache. Same frozen 66-file
corpus: 1,114,088 bytes, 97,485 nodes, 197,393 tokens. This implementation snapshot
emits 4,392 diagnostics and 3,995 proposed fixes; the prepared control-flow rule
emits 1,042 diagnostics. Historical counts belong to different snapshots.

Two balanced ABBA/BAAB screening blocks, then eight predeclared four-process
confirmation blocks. Each process has three warmups and eight measured batches
of four complete corpora. Percent changes are geometric within-block B/A ratios;
intervals are exploratory 95% bootstraps over the eight blocks, not independent
batch observations. Preserve raw schedules, all observations, and failed attempts.

Prepared control excludes parse/context prerequisites. Complete lint includes
parsing, context construction, every rule, diagnostics, allocation, and teardown.
The fix-collecting boundary collects proposals; it does not apply or format them.
Directory lint also includes warm OS reads. These are not cold-disk/startup times.
The frozen harness needed the current `Fixes.apply` API's new third argument;
`format=false` preserves its existing apply-only behavior identically in both
arms. Production APIs were not changed to accommodate the harness.

### Screen and confirmation

The screen removed 54.32% of rule instructions and 22.43% of rule cycles;
complete-lint cycles fell 3.78%. Wall times were severely disturbed, including
blocks with apparent regressions. Retain those results, but do not use that small
screen as the final latency estimate.

Eight-block confirmation, with 100% hardware-counter coverage throughout:

| Boundary | Instructions | Cycles (interval) | Wall time (interval) |
| --- | ---: | ---: | ---: |
| Prepared control-flow rule | −54.322% | −23.308% (−23.819%, −22.672%) | −24.072% (−25.781%, −22.420%) |
| Complete lint | −9.500% | −2.327% (−3.116%, −1.447%) | −2.889% (−4.943%, −0.009%) |
| Complete lint + fixes | −9.001% | −2.551% (−3.198%, −1.743%) | −2.982% (−4.005%, −1.900%) |
| Directory lint | −9.319% | −2.608% (−3.202%, −1.867%) | −2.660% (−4.324%, −0.970%) |

Raw complete-lint medians: **70.502 M → 68.342 M cycles**, **20.561 → 19.957 ms**,
and **182.357 M → 165.034 M instructions**. These medians are descriptive, not
the paired-block estimators above. About 17.324 M instructions disappear per
no-fix corpus. Rule cycle medians are 9.717 M → 7.430 M. All eight complete-lint
blocks improve in cycles.

The trade-off is real: rule branch misses rise about 53.68% (66.4 K → 102.0 K),
and complete-lint misses rise 5.40%. Instruction deletion is much larger than
cycle reduction; do not present them as interchangeable or attribute the entire
gap to branch misses without further measurements. The observed net cycle win
includes the new dispatch cost.

Host limitations remain: 125/128 stage observations report thermal-throttle
increases, 123 report sibling activity, and effective frequency spans 3.06–3.56
GHz. No frequency-policy changes occurred. The complete-lint wall interval barely
excludes zero. Prefer the repeatable cycle/instruction evidence over a precise
wall-time promise, and do not compare absolute timings directly with older,
concurrently different linter snapshots.

### Generated code and reprofile

Native disassembly shows range and bit-test dispatch on the loaded tag before
AST aggregate-copy setup and decoder calls. Unrelated nodes bypass that setup;
matching nodes still use standard decoders with safety checks. The existing
propagation worklist remains separate. This is not a branch-free optimization.

Before/after profiles each cover 600 complete no-fix corpora after ten warmups,
with acknowledged perf enable/disable windows excluding preparation and teardown.
There are 12,761 / 12,193 samples, zero lost samples, and no missing callchains.
Tail-call ownership uses each binary's own disassembled Linter return addresses.

| Disjoint owner | Before | After |
| --- | ---: | ---: |
| AST construction | 43.29% | 45.44% |
| Statement paragraphs | 18.33% | 18.39% |
| Top-down declarations | 12.47% | 12.81% |
| Control flow | 13.73% | 11.11% |

Sampled context-classification share falls from 6.31% to 4.64% of whole lint.
These are attribution figures, not another timing verdict; larger shares for
untouched components do not prove regressions. Instruction attribution across
inlined functions is approximate. Parsing remains the largest overall owner;
statement paragraphs are now the largest rule owner on this snapshot.

### Correctness and integration

Exact comparison passes on **1,479 inputs in Debug and ReleaseSafe**: line/token
indexes, rendered diagnostics, ordered edits, replacement bytes, fixed output,
and applied/skipped counts. These include existing corpus/fixtures/edge inputs,
the previous 1,324 temporary generated cases, and 43 additional temporary cases
covering all ten dispatch tags, block sizes/terminators, inline switch arms,
nested conditionals, loops, fallbacks, wrappers, captures, and LF/CRLF.

The integrated rule matches the measured candidate byte-for-byte. The change is
+53/−37 lines, net +16, within one loop. Every existing comment is preserved,
with indentation changed to follow its owning operation. No helpers, persistent
metadata, production test files, fixture additions, or snapshot rewrites.

Integrated checks pass through `agent-run`: full Debug and ReleaseSafe project
tests, rule/native formatting, and the ReleaseSafe install build. Unrelated
concurrent changes remain intact.

Evidence:
`/tmp/experiments/voiced-lint/baseline-20260908/evidence/20260909T023141Z-control-dispatch`.
Retained artifacts include the candidate patch, source/binary hashes, generators,
validators, census, raw schedules/invocations/counters, host conditions, initial
harness API failures, profiles/stacks/disassembly, analysis scripts, and check
records. Frozen baseline/candidate implementations remain under the same
experiment base in `control-dispatch-base` and `control-dispatch-switch`.
