# How Base.en and Small.en turn sound into text

This walkthrough compares two of the three English Whisper models supported by
this runtime. Each stage starts with its underlying idea, then shows what
actually happens in **Base.en** and **Small.en**. Medium.en uses the same
operations with 1,024-wide representations, 16 attention heads, and 24 encoder
and decoder layers.

We follow a full **30-second audio input**. Shorter encoder inputs change the
number of audio positions, not the learned weight dimensions. Shapes below are
logical: a row represents a position and a column represents a feature, unless
stated otherwise. Packing and transposing arrays in memory do not change these
relationships.

## 1. What is the model, and what is produced by running it?

**Theory:** a trained model is an architecture together with learned numerical
parameters; inference applies those fixed parameters to new inputs.

Both models use the same kinds of operations, but Small.en has wider
representations and more layers:

| Architecture property | Base.en | Small.en |
|---|---:|---:|
| Mel frequency bands | 80 | 80 |
| Encoder positions at full context | 1,500 | 1,500 |
| Features per encoder/decoder position | 512 | 768 |
| Encoder layers | 6 | 12 |
| Decoder layers | 6 | 12 |
| Attention heads per attention operation | 8 | 12 |
| Features per attention head | 64 | 64 |
| Feed-forward expansion width | 2,048 | 3,072 |
| Vocabulary entries in this runtime's model representation | 51,864 | 51,864 |

“Wider” means more numbers describe each position—not more recorded seconds or
more output words. Each layer has its own learned parameters; Small.en does not
just run Base.en's layers twice.

```text
TRAINING — happened elsewhere

Audio + correct text → predictions → measure error → adjust parameters
                                           │
                                  repeat over examples
                                           │
                                           ▼
                                 trained parameters
                                           │
INFERENCE                                  ▼

New audio ───────────────► fixed model computations ─────────► text
                                  │
                                  ▼
                         input-dependent results
```

The `Model` in the source holds fixed, inference-ready data. The encoder and
decoder code implement the operations that use it. A `Runtime` holds the working
state for those operations. Transcription does not update the trained
parameters.

The distinction is visible in a single Base.en projection:

```text
Learned projection weights:    512 × 2,048 numbers    same for every recording
Input to the projection:    1,500 ×   512 numbers    from this recording
Projection result:          1,500 × 2,048 numbers    calculated for this recording
```

For Small.en, the corresponding dimensions are `768 × 3,072`, `1,500 × 768`, and
`1,500 × 3,072`. The first array is model parameter data; the other two are
**activations**, the numerical representations produced while processing an
input. Their individual features generally do not have simple labels such as
“vowel” or “pitch.”

Not everything fixed is learned, and not everything derived is temporary:

| Kind of data | Examples here | Origin |
|---|---|---|
| Learned parameters | Projection weights, biases, embeddings, normalization scale/shift | Training |
| Fixed supporting data | Vocabulary mapping, window/Fourier/Mel tables | Token scheme or mathematical definitions |
| Prepared model data | Packed weights, weight-quantization scales and compensation | Derived from parameters for execution |
| Results for this input | Mel features, encoder states, queries/keys/values, token scores | Inference |
| Retained results for this input | Decoder K/V caches | Earlier inference steps |

For example, Mel **filter coefficients** stay fixed, but Mel **features** change
with the recording. Packing a weight matrix or filling a cache is not training.

*Source:* [Model.zig](../src/inference/Model.zig),
[Runtime.object.zig](../src/inference/Runtime.object.zig).

## 2. What a matrix multiplication actually does

**Theory:** each output feature is the sum of input features multiplied by their
corresponding weights; the same weights are used at every position.

The real feed-forward expansion has these dimensions:

```text
                INPUT                   WEIGHTS                  OUTPUT
             positions ×              input features ×        positions ×
             input features           output features         output features

Base.en      1,500 × 512         ×       512 × 2,048       →    1,500 × 2,048
Small.en     1,500 × 768         ×       768 × 3,072       →    1,500 × 3,072
```

If written as `X: N × D`, `W: D × F`, and `XW: N × F`, then **F is the number of
output columns**: 2,048 here for Base.en and 3,072 for Small.en. Multiplication
sums over the matching input-feature dimension, D; it does not sum across
positions.

### A complete calculation, with every term visible

To see the arithmetic without printing hundreds of learned coefficients, reduce
just this illustration to two positions, three input features, and two output
features. These numbers are chosen for clarity, not taken from either model.

```text
X — input values                     W — example weights

             input feature                         output feature
              0   1   2                             0   1
position 0  [ 3   2   1 ]             input feature 0 [ 2   1 ]
position 1  [ 1   0   2 ]             input feature 1 [ 4   3 ]
                                     input feature 2 [ 5   2 ]

Shape: 2 × 3                         Shape: 3 × 2
```

For `Y[0,0]`, take **row 0 of X**, `[3, 2, 1]`, and **column 0 of W**,
`[2, 4, 5]`. Match entries by their input-feature index:

```text
X[0,0] × W[0,0]     3 × 2
X[0,1] × W[1,0]     2 × 4
X[0,2] × W[2,0]     1 × 5
```

All four output cells are calculated this way. Here is every multiplication in
every sum, including products involving zero:

```text
Row 0 of X · column 0 of W:
Y[0,0] = 3×2 + 2×4 + 1×5 = 6 + 8 + 5 = 19

Row 0 of X · column 1 of W:
Y[0,1] = 3×1 + 2×3 + 1×2 = 3 + 6 + 2 = 11

Row 1 of X · column 0 of W:
Y[1,0] = 1×2 + 0×4 + 2×5 = 2 + 0 + 10 = 12

Row 1 of X · column 1 of W:
Y[1,1] = 1×1 + 0×3 + 2×2 = 1 + 0 + 4 = 5
```

The result is:

```text
Y = XW

             output feature
               0    1
position 0  [ 19   11 ]
position 1  [ 12    5 ]

Shape: 2 × 2
```

This is not element-by-element multiplication. Each output cell combines a
whole input row with a whole weight column. The output preserves the two
positions while changing the feature width from three to two.

Back in the real models, **each Base.en expansion output cell sums 512
products**, and **each Small.en expansion output cell sums 768 products**.
There are 2,048 or 3,072 such output cells per position. The arithmetic rule is
identical to the fully expanded example. Projection layers may additionally add
a learned output offset; the example above isolates the matrix product.

*Source:* [linear.zig](../src/inference/linear.zig).

## 3. From 30 seconds of sound to audio features

**Theory:** log-Mel extraction describes how energy in frequency bands changes
over time, using fixed signal-processing operations rather than learned weights.

The starting calculation is the same for both models:

```text
30 seconds × 16,000 samples/second
                  │
                  ▼
          480,000 waveform samples
                  │
                  ▼
     overlapping 400-sample windows
        advancing 160 samples at a time
                  │
                  ▼
     frequency analysis → power spectrum
                  │
                  ▼
       80 Mel bands → log + normalization
                  │
                  ▼
         3,000 frames × 80 features
```

A sample measures waveform amplitude. Frequency content comes from how samples
vary over a window. The 400-sample windows span 25 ms, and their 160-sample steps
advance by 10 ms. The resulting 3,000 frames are overlapping observations, not
3,000 separate recognized sounds.

Mel filters combine frequency bins into overlapping bands with spacing that
reflects aspects of pitch perception. Logarithms compress the range of energies.
Neither Base.en nor Small.en has recognized words at this point, and Small.en
gets the same 80-band input rather than a higher-resolution spectrogram.

The diagrams view each frame as a row. The source represents this feature matrix
with bands first (`80 × 3,000`); that is a storage orientation, not a different
set of features.

*Source:* [log_mel.zig](../src/inference/log_mel.zig).

## 4. From Mel features to encoder positions

**Theory:** a convolution applies learned weighted sums to neighboring frames,
extracting local patterns with the same weights at each time position.

```text
                         Base.en                    Small.en

Mel features             3,000 × 80                  3,000 × 80
                              │                          │
                         convolution + GELU         convolution + GELU
                              ▼                          ▼
                         3,000 × 512                 3,000 × 768
                              │                          │
                         strided convolution        strided convolution
                         + GELU                     + GELU
                              ▼                          ▼
Encoder positions        1,500 × 512                 1,500 × 768
                              │                          │
                         add position information   add position information
```

The first convolution uses three neighboring Mel frames. Each output feature is
therefore a weighted sum of `3 × 80 = 240` input values. It produces 512 features
per position in Base.en and 768 in Small.en.

The second convolution combines three neighboring learned vectors: 1,536 input
values in Base.en or 2,304 in Small.en for each weighted sum. Its stride reduces
the position count from 3,000 to 1,500 while keeping the feature width unchanged.
GELU is a smooth nonlinear transformation applied to the projection results.

Position information gives the network a way to distinguish earlier and later
positions. The result is still numerical audio data: one row does not mean one
word. The 1,500 rows are encoder positions, distinct from waveform samples,
feature frames, and the eventual text tokens.

*Source:* [encoder.zig](../src/inference/encoder.zig),
[layout.zig](../src/packed_model/layout.zig) for convolution tensor shapes.

## 5. Inside one encoder layer

Each layer follows the same sequence, preserving its input/output shape:

```text
Input ─────┬──────────────────────────┐
           ▼                          │
       LayerNorm                      │
           ▼                          │
       Self-attention                 │
           ▼                          │
       Output projection              │
           ▼                          │
           + ◄────────────────────────┘
           │
           ├──────────────────────────┐
           ▼                          │
       LayerNorm                      │
           ▼                          │
       Feed-forward network           │
           ▼                          │
           + ◄────────────────────────┘
           │
           ▼
         Output

Base.en:   1,500 × 512 in → 1,500 × 512 out; repeat through 6 layers
Small.en:  1,500 × 768 in → 1,500 × 768 out; repeat through 12 layers
```

The long paths add the incoming state back to the calculated update. These are
**residual connections**: a layer refines an existing representation rather than
having to replace it completely.

### Normalize each position

**Theory:** LayerNorm calculates a position's feature mean and variance, then
normalizes those values and applies learned scale and shift parameters.

| At one position | Base.en | Small.en |
|---|---|---|
| Values used to calculate the mean/variance | 512 | 768 |
| Learned scale vector, gamma | 512 values | 768 values |
| Learned shift vector, beta | 512 values | 768 values |
| Output feature width | 512 | 768 |

This happens independently for all 1,500 rows. The statistics change with the
input; gamma and beta stay fixed. The normalization before attention and the
normalization before the FFN have their own learned parameters.

*Source:* [normalization.zig](../src/inference/normalization.zig), [encoder.zig](../src/inference/encoder.zig).

### Project into queries, keys, and values

**Theory:** learned projections produce queries for matching, keys to match
against, and values containing the information to combine.

```text
Base.en

Normalized state      Learned Q/K/V weights       Calculated Q/K/V
1,500 × 512       ×        512 × 1,536         →     1,500 × 1,536
                                                        │ split columns
                                                        ├─ Q: 1,500 × 512
                                                        ├─ K: 1,500 × 512
                                                        └─ V: 1,500 × 512

Small.en

Normalized state      Learned Q/K/V weights       Calculated Q/K/V
1,500 × 768       ×        768 × 2,304         →     1,500 × 2,304
                                                        │ split columns
                                                        ├─ Q: 1,500 × 768
                                                        ├─ K: 1,500 × 768
                                                        └─ V: 1,500 × 768
```

Each calculated cell sums 512 products in Base.en or 768 in Small.en, using the
row-times-column rule from section 2. The combined result contains three
representations, not a wider replacement for the encoder state.

**The projection matrix is model parameter data; Q, K, and V are results for
this input.** A new recording changes the latter, not the former.

### Compare positions and combine their information

**Theory:** attention compares queries with keys and turns the comparisons into
input-dependent weights for mixing values.

Base.en splits each Q/K/V representation into **8 heads of 64 features**;
Small.en splits it into **12 heads of 64 features**. Each head performs the same
sized calculation for this audio length:

```text
ONE HEAD — same dimensions in Base.en and Small.en

Q                        K transposed                Comparison scores
1,500 × 64         ×       64 × 1,500             →     1,500 × 1,500

Each score is a sum of 64 query-component × key-component products.
The transpose makes each original key row available as a column.

                         divide scores by √64 = 8
                                      │
                                      ▼
                         softmax across each row
                                      │
                                      ▼
Mixing weights           V                           Head result
1,500 × 1,500      ×      1,500 × 64              →      1,500 × 64

Each result cell is a sum of 1,500 mixing-weight × value-component products.
```

A score row belongs to one query position; its 1,500 columns refer to possible
source positions. **Softmax** converts that row into shares summing to one, with
larger scores receiving larger shares. The second multiplication gathers a
weighted mixture of source information for that query.

These “attention weights” are calculated from the input, unlike the learned
projection weights. They are not probabilities of output words. The full score
matrix describes the logical calculation; it need not be stored as one array.

The heads' results are joined and projected back into the encoder state:

```text
Base.en:   8 × 64 features → 1,500 × 512 → projection by 512 × 512
Small.en: 12 × 64 features → 1,500 × 768 → projection by 768 × 768
```

Each output-projection cell sums 512 or 768 products respectively. Its result is
added to the layer's incoming state, leaving the same number of positions and
features. Encoder self-attention can use the entire supplied audio sequence,
including later positions.

*Source:* [attention.zig](../src/inference/attention.zig), [encoder.zig](../src/inference/encoder.zig),
[Model.zig](../src/inference/Model.zig) for the learned projection tensors.

### Transform the features within each position

**Theory:** the feed-forward network expands each position's features, applies a
nonlinear function, and projects back to model width.

```text
Base.en                            Small.en

1,500 × 512                        1,500 × 768
    │ multiply by 512 × 2,048          │ multiply by 768 × 3,072
    ▼                                  ▼
1,500 × 2,048                      1,500 × 3,072
    │ GELU                             │ GELU
    ▼                                  ▼
1,500 × 2,048                      1,500 × 3,072
    │ multiply by 2,048 × 512          │ multiply by 3,072 × 768
    ▼                                  ▼
1,500 × 512                        1,500 × 768
```

The multiplier matrices contain learned weights. Expansion sums 512 products per
output cell in Base.en and 768 in Small.en; contraction sums 2,048 and 3,072
respectively. GELU changes the intermediate values without changing their shape.
Its nonlinearity prevents the two projections from being equivalent to just one
projection.

Attention mixes information **between positions**. The FFN transforms features
**within each position**, using the same weights at each row. Its result is added
to the preceding state through the second residual connection.

After all 6 or 12 encoder layers and a final normalization, the encoded audio
still has shape `1,500 × 512` or `1,500 × 768`. Its values now incorporate context
from across the recording. They are not yet text.

*Source:* [encoder.zig](../src/inference/encoder.zig), [linear.zig](../src/inference/linear.zig).

## 6. From encoded audio to one decoder step

### Turn a token ID into a learned vector

**Theory:** a token embedding is a learned feature vector selected by an integer
token ID; it is distinct from that token's spelling or text bytes.

| | Base.en | Small.en |
|---|---|---|
| Learned embedding table | 51,864 × 512 | 51,864 × 768 |
| One selected token's vector | 1 × 512 | 1 × 768 |
| After adding its position information | 1 × 512 | 1 × 768 |

The vocabulary mapping associates IDs with text bytes or control-token roles.
The embedding table associates those same IDs with learned numerical vectors.
A token may cover a word, part of a word, punctuation, or bytes; it need not
correspond to one encoder position.

Setup converts the upstream GPT-2 vocabulary into concatenated decoded token
bytes and a token-offset table inside the same `.voiced` file as the weights.
The runtime copies each selected byte span directly and validates the completed
transcript as UTF-8; it does not parse or translate vocabulary text while the
model is resident.

The decoder starts with control tokens describing the task. After that, each
step processes the next selected token with its accumulated context.

### Self-attention uses the text context

**Theory:** causal self-attention lets the current text position use itself and
preceding text positions, but not future ones.

Consider a step with **20 processed text-side positions**, including the current
position and prompt tokens. This is a point in a growing sequence, not an
architecture limit.

```text
                                 Base.en                Small.en
Current state                    1 × 512                1 × 768
Self-attention Q/K/V projection   512 × 1,536             768 × 2,304
Calculated Q/K/V                  1 × 1,536              1 × 2,304
Heads                            8                      12

Per head, in either model:

Current query          Cached keys transposed        Scores over text
1 × 64             ×          64 × 20            →         1 × 20
                                                           │
                                                  divide by 8 + softmax
                                                           ▼
Mixing weights         Cached values                 Head result
1 × 20             ×          20 × 64            →         1 × 64
```

Each score sums 64 products. Each output feature sums 20 products, one from each
available text-side position. The heads are joined and output-projected back to
512 or 768 features, then added to the incoming state.

### Cross-attention uses the audio context

**Theory:** cross-attention uses a query from the current text state to gather
information from the encoded audio.

Each decoder layer has its own learned projection of the audio into K/V:

```text
Base.en:   [1,500 × 512] × [512 × 1,024] → [1,500 × 1,024]
                                        split into K and V: each 1,500 × 512

Small.en:  [1,500 × 768] × [768 × 1,536] → [1,500 × 1,536]
                                        split into K and V: each 1,500 × 768
```

These output cells sum 512 or 768 products respectively. The audio does not
change between generated tokens, so these results can be reused. The current
text state supplies a new query through a `512 × 512` projection in Base.en or a
`768 × 768` projection in Small.en.

After splitting into heads, both models perform this calculation per head:

```text
Current text query     Audio keys transposed          Scores over audio
1 × 64             ×        64 × 1,500           →        1 × 1,500
                                                           │
                                                  divide by 8 + softmax
                                                           ▼
Mixing weights         Audio values                  Head result
1 × 1,500          ×        1,500 × 64           →          1 × 64
```

Each audio score sums 64 products. Each result feature sums 1,500 weighted value
components. Compared with self-attention at the example step, the source length
is 1,500 audio positions instead of 20 text positions. Base.en performs this over
8 heads; Small.en over 12.

The joined result is output-projected and added back to the text state. Each
layer then applies the same type of FFN as the encoder, but to **one current text
row**:

```text
Base.en:   1 × 512 → 1 × 2,048 → GELU → 1 × 512
Small.en:  1 × 768 → 1 × 3,072 → GELU → 1 × 768
```

Normalization, residual additions, and the feature widths work as in the encoder.
Base.en runs through 6 decoder layers; Small.en through 12. Each layer has its
own parameters, including distinct self- and cross-attention projections.

*Source:* [decoder.zig](../src/inference/decoder.zig), [attention.zig](../src/inference/attention.zig).

## 7. What exactly is cached?

**Theory:** a K/V cache retains already-calculated keys and values so later
queries can use them without repeating the projections.

At the 20-position step above, each decoder layer has these logical caches:

| Per decoder layer | Base.en | Small.en | Reuse |
|---|---|---|---|
| Cross-attention K | 1,500 × 512 | 1,500 × 768 | Same audio throughout decoding |
| Cross-attention V | 1,500 × 512 | 1,500 × 768 | Same audio throughout decoding |
| Self-attention K | 20 × 512 | 20 × 768 | Earlier text plus current position |
| Self-attention V | 20 × 512 | 20 × 768 | Earlier text plus current position |

At the next step, self-attention gets a 21st K row and V row. Cross-attention
keeps using the same audio K/V. Causality means earlier text positions do not
change their representations in response to later tokens, so their K/V remains
reusable.

```text
                           Base.en                 Small.en
Self K or V, this step      20 × 512                20 × 768
Self K or V, next step      21 × 512                21 × 768

Cross K or V, both steps    1,500 × 512             1,500 × 768
```

These caches are duplicated **by layer**, not shared as one universal table,
because each layer's projection weights differ. Their contents depend on this
recording and text sequence. They are not new trained parameters or knowledge
that the model keeps learning between recordings.

The dimensions above count live logical values, not allocated capacity or a
promise about cache precision and byte layout.

*Source:* [decoder.zig](../src/inference/decoder.zig), [Runtime.object.zig](../src/inference/Runtime.object.zig).

## 8. From a decoder vector to the next token

**Theory:** the decoder converts its final feature vector into a score for every
vocabulary entry, then a selection rule chooses the next token.

After final normalization, the runtime reuses the learned embedding matrix for
the output projection:

```text
               CURRENT STATE       EMBEDDING TABLE TRANSPOSED       LOGITS

Base.en         1 × 512        ×          512 × 51,864         →    1 × 51,864
Small.en        1 × 768        ×          768 × 51,864         →    1 × 51,864
```

Each Base.en vocabulary score sums 512 state-component × embedding-component
products. Each Small.en score sums 768. The two models produce the same number
of scores, but arrive at them through different learned parameters and feature
representations.

The scores are **logits**, not probabilities. Softmax can turn them into a
distribution over next-token candidates. This is different from attention's
softmax over source positions. Here, the alternatives are the 51,864 vocabulary
entries, not the 1,500 audio positions or the 20 text-side positions.

This runtime excludes disallowed tokens and makes a greedy choice: select the
highest-scoring remaining token. That ID becomes the next input, and the process
repeats until an end marker or configured limit. Vocabulary lookup converts the
selected text-token IDs into output bytes; the spelling is not what the output
matrix multiplication compares against.

Every prediction depends on both the audio and the preceding tokens. The decoder
is therefore not assigning one word independently to each time slice. Language
patterns learned during training help resolve unclear speech, but can also
produce plausible text when audio evidence is weak. A high predicted token
probability is not an independent guarantee that the word was spoken.

*Source:* [decoder.zig](../src/inference/decoder.zig), [Runtime.object.zig](../src/inference/Runtime.object.zig).

## 9. The learned model versus its execution representation

**Theory:** quantization approximates numerical values, packing rearranges their
storage, and caching retains computed results—three different operations.

For the Base.en FFN expansion, the logical weight matrix remains `512 × 2,048`;
for Small.en it remains `768 × 3,072`. Preparing these matrices for execution can
change their precision and physical arrangement without changing which input
and output features each coefficient connects.

Weight-quantization scales and arithmetic compensation are derived from the
fixed parameters. Activations can also be quantized, but their values and scales
then depend on the current input. Neither process trains new parameters here.
Packing alone changes element order, not their numerical precision.

Similarly, the `1,500 × 1,500` per-head attention calculation does not require a
permanent array of that size. Work can be evaluated in pieces, fused with other
operations, or placed in storage reused after earlier results are no longer
needed. Workers cooperate on these calculations; they are not independently
trained models.

The shapes in this walkthrough come from the Base.en and Small.en architectures,
not tile sizes, worker counts, or byte offsets. A different kernel need not
change the explanation. `Model.Dimensions` remains the source of truth for the
architecture dimensions.

*Source:* [layout.zig](../src/packed_model/layout.zig),
[writer.zig](../src/packed_model/writer.zig),
[vnni_weight.zig](../src/inference/vnni_weight.zig),
[Runtime.object.zig](../src/inference/Runtime.object.zig), and [Scheduler.zig](../src/inference/Scheduler.zig).
