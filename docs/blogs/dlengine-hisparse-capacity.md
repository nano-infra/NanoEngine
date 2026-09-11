## SGLang HiSparse Capacity Model

SGLang HiSparse splits the KV / Indexer cache into three tiers: **logical** (scheduling semantics), **device hot buffer** (a small swap-in window on the GPU), and **host cold tier** (the full KV in CPU pinned memory). This post builds a quantitative capacity model for SGLang's **ratio-driven allocation policy**: what "capacity" means, which inequalities bound it, where the GPU memory actually goes, and what configurations are feasible for two concrete models (GLM5.1 and GLM5.2) on two H100 deployments.

The NSA / FP8 KV / Indexer byte layouts are covered in [DLEngine - NSA](dlengine-nsa.md). DLEngine's buffer-first allocator is a different policy and is not modeled here.

### 1. Background and Assumptions

Deploying SGLang HiSparse boils down to choosing three knobs — the per-sequence device buffer size (`hisparse_device_buffer_size`), the host/device logical ratio (`hisparse_host_to_device_ratio`), and the maximum concurrency (`max_num_seqs`) — under a fixed GPU memory budget. The capacity model in this post is built on two assumptions:

1. **The Indexer cache is fully resident in GPU memory.** HiSparse only splits the MLA KV between the host cold tier and the device hot buffer; the Indexer cache is provisioned on the GPU for the complete logical token space. Some models amortize this cost by sharing one Indexer cache across multiple model layers, captured by the layer-sharing factor $`R_{Share} = N_{layer} / N_{layer,indexer}`$ (GLM5.1 keeps a per-layer Indexer cache, so $`R_{Share}=1`$; GLM5.2 shares the Indexer across layers at $`78/21 \approx 3.714`$). The Indexer cost per hot token slot in the GPU memory model is therefore $`R_{Host} \cdot B_{T,Indexer} / R_{Share}`$.
2. **No prefix cache.** The capacity calculation assumes every logical token independently occupies KV / Indexer space, with no deduplication or reuse of shared prefixes. With prefix caching enabled, the effective serviceable capacity additionally depends on workload prefix overlap and hit rate, which is out of scope here (see the TODO in Section 8).

### 2. Symbols

| Symbol | Definition | Typical value / source |
| --- | --- | --- |
| $`N_{T,Buffer}`$ | Device hot capacity (per-seq): token slots resident on the GPU hot tier per sequence; the swap-in window of sparse decode. Short sequences take the fast path when $`L_{seq}\le N_{T,Buffer}`$. | $`6144`$ — `hisparse_device_buffer_size`; validated by the ESS paper (arXiv) and the SGLang HiSparse default |
| $`B^T_{Buffer}`$ | Device hot capacity (per-batch): total GPU hot token slots of a decode batch | $`N_{bs,max}N_{T,Buffer}`$ |
| $`R_{Host}`$ | SGLang logical-capacity expansion factor: the number of logical Indexer tokens provisioned per GPU MLA hot slot, equivalently $`C_{Batch}/B^T_{Buffer}`$ | `hisparse_host_to_device_ratio`; $`R_{Host}\ge1`$ |
| $`C_{host,seq}`$ | Host logical capacity (per-seq): logical tokens one sequence can cover in the host namespace | $`R_{Host}N_{T,Buffer}`$; e.g. $`2\times6144=12{,}288`$ tokens |
| $`C=C_{Batch}`$ | Decode-batch logical capacity: maximum aggregate tokens one batch can serve | Derived in Lemma 1 |
| $`N_{Topk}`$ | Sparse hard floor: top-k entries one sparse attention step must hold; $`N_{T,Buffer}`$ may not go below it | $`2048`$ — `index_topk` |
| $`L_{max,model}`$ | Model context ceiling: maximum logical tokens per request | GLM5.1: $`262{,}144`$ (256K); GLM5.2: $`1{,}048{,}576`$ (1M) |
| $`R_{Share}`$ | Indexer layer-sharing factor $`N_{layer}/N_{layer,indexer}`$: how many model layers share one Indexer cache | GLM5.1: $`1`$ (per-layer); GLM5.2: $`78/21\approx3.714`$ |
| $`B_{T,Indexer}`$ | Indexer cache bytes per logical token per (Indexer) layer | $`132\ \mathrm{B}`$ |
| $`B_{T,MLA}`$ | FP8 MLA KV bytes per token per layer | $`656\ \mathrm{B}`$ |
| $`N_{layer}`$ | Model layer count | $`78`$ (both GLM5.1 and GLM5.2) |
| $`N_{bs,max}`$ | Maximum concurrent sequences | $`8`$ — `max_num_seqs` |
| $`M_{Available}`$ | Nominal GPU HBM | H100: $`77.47\ \mathrm{GB}`$; H200: $`140\ \mathrm{GB}`$ |
| $`M_{weights}`$ | Per-GPU resident weight memory | DP16EP16: $`64.52\ \mathrm{GB}`$; DP32EP32: $`43.42\ \mathrm{GB}`$ |
| $`F`$ | Fraction of GPU memory granted to KV / cache | $`0.88`$ (DP16EP16); $`0.82`$ (DP32EP32) |
| $`M_{cache}`$ | HBM available to the HiSparse Buffer and Indexer | $`M_{Available}F-M_{weights}`$ |
| $`M_{Host,Available}`$ | Host memory available to the worker's cold MLA KV | Deployment parameter |

> Note: $`6144`$ is the value validated by the ESS paper and adopted as the SGLang HiSparse default, and is what this post standardizes on.

### 3. Capacity Derivation

#### 3.1 Define the host/device scaling factor

SGLang defines $`R_{Host}`$ as the configured **logical-capacity expansion factor** between the full logical Indexer namespace and the GPU MLA hot Buffer:

$$
\boxed{
\begin{aligned}
R_{Host}
&:=\frac{C_{Batch}}{B^T_{Buffer}}
=\frac{\mathtt{size\_full}}{\mathtt{size\_device}},
\qquad R_{Host}\ge1, \\
C=C_{Batch}&=R_{Host}B^T_{Buffer}.
\end{aligned}
}
\tag{3.1}
$$

This is a ratio of **token-slot counts**, not a byte ratio between host RAM and HBM. It is an input to the SGLang allocator rather than a quantity measured after allocation. Equivalently, SGLang uses $`\mathtt{size\_full}=\mathtt{size\_device}\cdot R_{Host}`$. Equation (3.1) defines the logical namespace; it is not a result inferred after memory allocation. The remaining derivation asks how large $`B^T_{Buffer}`$ can be under the HBM budget and, through Equation (3.1), how large $`C_{Batch}`$ can be.

#### 3.2 Decompose the HBM budget

The MLA device Buffer stores only the hot token slots, whereas the fully resident Indexer covers all $`C_{Batch}`$ logical tokens. Their HBM costs are:

$$
\begin{aligned}
M_{Buffer}
&=N_{layer}B^T_{Buffer}B_{T,MLA}, \\
M_{Indexer}
&=N_{layer}\frac{C_{Batch}B_{T,Indexer}}{R_{Share}}.
\end{aligned}
\tag{3.2}
$$

Therefore the fundamental SGLang HBM budget is:

$$
\boxed{
M_{cache}
\ge
N_{layer}B^T_{Buffer}B_{T,MLA}
+N_{layer}\frac{C_{Batch}B_{T,Indexer}}{R_{Share}}
}
\tag{3.3}
$$

Substituting the scaling definition in Equation (3.1) gives:

$$
M_{cache}
\ge
B^T_{Buffer}N_{layer}
\left(
B_{T,MLA}+\frac{R_{Host}B_{T,Indexer}}{R_{Share}}
\right).
\tag{3.4}
$$

This form also explains the effective HBM cost of one hot slot:

$$
m_{slot}=N_{layer}
\left(
B_{T,MLA}+\frac{R_{Host}B_{T,Indexer}}{R_{Share}}
\right)
\tag{3.5}
$$

#### 3.3 Solve for the worker-wide Buffer

With saturated HBM, the worker-wide Buffer is:

$$
B^T_{Buffer}
=\frac{M_{cache}}{m_{slot}}
=\frac{M_{cache}}
{N_{layer}\left(
B_{T,MLA}+\dfrac{R_{Host}B_{T,Indexer}}{R_{Share}}
\right)}
\tag{3.6}
$$

For an undersubscribed pool, the equalities in Equation (3.6) become upper bounds. The figures use the saturated-HBM ceiling.

#### 3.4 Batch Capacity

> **Lemma 1 (HiSparse Batch Capacity).** Under the assumptions above and with saturated HBM, the aggregate logical token capacity of one decode batch is:

$$
\boxed{
\begin{aligned}
C=C_{Batch}
&=R_{Host}B^T_{Buffer} \\
&=\frac{M_{cache}R_{Host}}
{N_{layer}\left(
B_{T,MLA}+\dfrac{R_{Host}B_{T,Indexer}}{R_{Share}}
\right)} \\
&=\frac{M_{cache}R_{Host}}{m_{slot}}
\end{aligned}}
\tag{3.7}
$$

**Proof.** Equation (3.1) first defines the logical capacity represented by a worker-wide Buffer. Substituting that definition into the explicit two-term HBM budget in Equation (3.3) gives Equation (3.4), and saturating the budget gives the Buffer ceiling in Equation (3.6). Substituting this ceiling into Equation (3.1) yields Equation (3.7).

Equation (3.7) is the central Capacity formula plotted in this post.

#### 3.5 Relationship to batch size

An equal split is only one way to partition the worker-wide Buffer pool:

$$
B^T_{Buffer}=N_{bs,max}N_{T,Buffer}
\tag{3.8}
$$

Substituting this relation into the scaling definition in Equation (3.1) gives:

$$
C_{Batch}=N_{bs,max}R_{Host}N_{T,Buffer}
=N_{bs,max}C_{host,seq}
\tag{3.9}
$$

Here $`C_{host,seq}=R_{Host}N_{T,Buffer}`$. Equations (3.8)–(3.9) describe only a static partition of the total Buffer. At fixed $`M_{cache}`$ and $`R_{Host}`$, the worker-wide Capacity in Lemma 1 is independent of $`N_{bs,max}`$.

### 4. Capacity Constraints

The model applies the following three constraints in order.

> **Constraint 1 (Sparse and GPU-HBM feasibility).** The per-sequence Buffer must simultaneously satisfy the sparse kernel's top-k floor and the GPU-HBM ceiling.

The sparse kernel first imposes a hard lower bound:

$$
N_{T,Buffer}\ge N_{Topk}
\tag{4.1}
$$

A sequence's device buffer must hold at least one full top-k selection, otherwise swap-in cannot serve the sparse kernel at all. With weights resident and concurrency saturated, HBM imposes the corresponding upper bound:

$$
N_{T,Buffer}
\le
\frac{M_{Available}F-M_{weights}}
{\left(\dfrac{R_{Host}B_{T,Indexer}}{R_{Share}}+B_{T,MLA}\right)
 N_{layer}N_{bs,max}}
\tag{4.2}
$$

- Numerator: GPU memory available to the HiSparse hot cache, $`M_{cache}=M_{Available}F-M_{weights}`$.
- Denominator: the effective byte cost of one hot token slot across all layers at maximum concurrency. Each slot pays the full MLA KV cost $`B_{T,MLA}`$, plus the Indexer cost of the $`R_{Host}`$ logical tokens it represents, amortized by layer sharing: $`R_{Host}B_{T,Indexer}/R_{Share}`$.

Combining Equations (4.1) and (4.2) gives the feasible interval for the device buffer:

$$
N_{Topk}
\le N_{T,Buffer}
\le
\frac{M_{cache}}
{\left(\dfrac{R_{Host}B_{T,Indexer}}{R_{Share}}+B_{T,MLA}\right)
 N_{layer}N_{bs,max}}
\tag{4.3}
$$

> **Constraint 2 (Model-length coverage).** The worker-wide Capacity must cover at least one model-length logical context.

Using $`C_{Batch}=R_{Host}B^T_{Buffer}`$, two bounds apply — but they have different natures and should not be read as one two-sided feasibility inequality.

**Coverage requirement.** The figures use one model-length worth of aggregate worker capacity as the target:

$$
C_{Batch}\ge L_{max,model}
\quad\Longleftrightarrow\quad
R_{Host}\ge \frac{L_{max,model}}{B^T_{Buffer}}
\tag{4.4}
$$

Under an equal static split, $`B^T_{Buffer}=N_{bs,max}N_{T,Buffer}`$ and each sequence receives only $`C_{Batch}/N_{bs,max}`$. Under the elastic worker-wide pool assumed by the figures, however, an admitted request may consume more than an equal share whenever other requests do not need it. Consequently, $`L_{max,model}`$ is an aggregate capacity target, not a claim that all $`N_{bs,max}`$ requests can simultaneously reach the model limit.

**No-waste upper bound.** Conversely, no batch of at most $`N_{bs,max}`$ sequences can address more than $`N_{bs,max}`$ model-length contexts:

$$
C_{Batch}\le N_{bs,max}L_{max,model}
\tag{4.5}
$$

This is not a hard feasibility constraint — exceeding it does not break anything — but the excess pinned host memory can never be referenced by any admissible batch.

> **Constraint 3 (Physical host-memory ceiling).** The Host cold tier must physically back every logical token covered by the worker.

The host must physically back the logical pool. Let $`M_{Host,Available}`$ be the memory budget available to the worker's pinned MLA KV pool. Because the Indexer remains on the GPU, the logical capacity must satisfy:

$$
M_{Host}
\approx
C_{Batch}N_{layer}B_{T,MLA}
\le M_{Host,Available}
\tag{4.6}
$$

Therefore:

$$
C_{Batch}\le C^{Host}_{Batch,max}
=\frac{M_{Host,Available}}
{N_{layer}B_{T,MLA}}
\tag{4.7}
$$

or, under an equal static per-sequence partition:

$$
C_{host,seq}
\le
\frac{M_{Host,Available}}
{N_{bs,max}N_{layer}B_{T,MLA}}
\tag{4.8}
$$

This is a hard physical-capacity bound, unlike the no-waste bound above. The figures in this post intentionally do not include it because no host-memory budget is fixed in the plotted configurations; they show only the GPU, sparse/top-k, and model-length bounds.

### 5. Where the GPU Memory Actually Goes

The constraints above are phrased in logical token counts. This section interprets the Buffer / Indexer decomposition already established by Equations (3.2)–(3.3), to see what the GPU memory is ultimately spent on and what raising $`R_{Host}`$, $`R_{Share}`$, or $`N_{bs,max}`$ each actually buys.

#### 5.1 Buffer vs. Indexer composition

Equation (3.2) already gives both components: the MLA Buffer covers only $`B^T_{Buffer}`$ GPU hot slots, while the Indexer covers the complete $`C_{Batch}`$ logical namespace. Equation (3.3) then adds them under the same HBM budget. This is exactly the decomposition shown in the figures; it is not a second capacity model.

Because this post assumes the Indexer cache is fully GPU-resident, the host cold tier only stores the complete logical MLA KV. The host memory for one full-length sequence should therefore be estimated as:

$$
M_{host,seq}
\approx
L_{max,model}N_{layer}B_{T,MLA}
\tag{5.1}
$$

without adding $`B_{T,Indexer}`$ on the host side. If a future implementation also keeps an Indexer replica on the host, add $`L_{max,model}N_{layer}B_{T,Indexer}`$ on top.

#### 5.2 The Indexer share is decided by just two ratios

The Indexer-to-Buffer memory ratio is:

$$
\frac{M_{Indexer}}{M_{Buffer}}=
\frac{R_{Host}B_{T,Indexer}}
{R_{Share}B_{T,MLA}}
\tag{5.2}
$$

so the Indexer's share of the HiSparse cache is:

$$
P_{Indexer}=\frac{R_{Host}B_{T,Indexer}}
{R_{Share}B_{T,MLA}+R_{Host}B_{T,Indexer}}
\tag{5.3}
$$

This share is independent of $`N_{T,Buffer}`$, $`N_{bs,max}`$, $`N_{layer}`$, and $`M_{cache}`$: scaling concurrency or the buffer inflates both components proportionally without changing their mix.

The break-even point where $`M_{Indexer}=M_{Buffer}`$ — beyond which the Indexer becomes the dominant memory cost — is:

$$
R^{\mathrm{50pct}}_{Host}=\frac{R_{Share}B_{T,MLA}}{B_{T,Indexer}}
\approx 4.97\,R_{Share}
\tag{5.4}
$$

using $`B_{T,MLA}=656\ \mathrm{B}`$ and $`B_{T,Indexer}=132\ \mathrm{B}`$. Instantiated for the two models:

| `R_Host` | `P_Indexer`, GLM5.1 (`R_Share=1`) | `P_Indexer`, GLM5.2 (`R_Share=3.714`) |
| ---: | ---: | ---: |
| 5 | 50.2% | 21.3% |
| 10 | 66.8% | 35.1% |
| 20 | 80.1% | 52.0% |
| 40 | 88.9% | 68.4% |

For GLM5.1 the Indexer already overtakes the Buffer at $`R_{Host}\approx5`$; GLM5.2's layer sharing pushes that break-even out to $`R_{Host}\approx18.5`$. In long-context configurations, the dominant memory problem quickly shifts from the MLA hot buffer to the full-resident Indexer cache.

#### 5.3 Does the buffer grow or shrink as concurrency rises?

It depends on what is held fixed.

If the configured $`N_{T,Buffer}`$ is fixed, both Buffer and Indexer memory grow linearly with concurrency:

$$
M_{HiSparse}\propto N_{bs,max}
\tag{5.5}
$$

If instead the memory budget $`M_{cache}`$ is fixed and the buffer uses its full allowance, then:

$$
N^{GPU}_{T,Buffer}=\frac{M_{cache}}
{N_{layer}N_{bs,max}
\left(
B_{T,MLA}
+\dfrac{R_{Host}B_{T,Indexer}}{R_{Share}}
\right)}
\tag{5.6}
$$

so the per-sequence buffer is inversely proportional to concurrency:

$$
N^{GPU}_{T,Buffer}\propto\frac{1}{N_{bs,max}}
\tag{5.7}
$$

but the batch-wide hot-slot ceiling:

$$
B^{T,GPU}_{Buffer}
=N_{bs,max}N^{GPU}_{T,Buffer}
=\frac{M_{cache}}
{N_{layer}
\left(
B_{T,MLA}
+\dfrac{R_{Host}B_{T,Indexer}}{R_{Share}}
\right)}
\tag{5.8}
$$

is independent of $`N_{bs,max}`$. In a memory-saturated deployment, raising concurrency does not create more hot slots — it only splits a fixed hot-slot budget across more sequences.

#### 5.4 Host ratio payoff has an Indexer ceiling

Substituting $`N^{GPU}_{T,Buffer}`$ into $`C_{Batch}=N_{bs,max}R_{Host}N_{T,Buffer}`$:

$$
C^{GPU}_{Batch}(R_{Host})=\frac{M_{cache}R_{Host}}
{N_{layer}
\left(
B_{T,MLA}
+\dfrac{R_{Host}B_{T,Indexer}}{R_{Share}}
\right)}
\tag{5.9}
$$

This expression contains no $`N_{bs,max}`$: in the idealized continuous-allocation model, adding concurrency only changes how capacity is divided among sequences, never the worker's total logical capacity.

As $`R_{Host}\rightarrow\infty`$:

$$
\lim_{R_{Host}\to\infty}C^{GPU}_{Batch}
=\frac{M_{cache}R_{Share}}
{N_{layer}B_{T,Indexer}}
\tag{5.10}
$$

This is the capacity ceiling imposed by the GPU-resident Indexer. Raising $`R_{Host}`$ drives the MLA device-buffer cost down, but the per-logical-token Indexer cost never goes away. For a fixed target capacity $`C_{Batch}`$, the two components can equivalently be written as:

$$
\begin{aligned}
M_{Buffer}
&=N_{layer}C_{Batch}\frac{B_{T,MLA}}{R_{Host}}, \\
M_{Indexer}
&=N_{layer}C_{Batch}\frac{B_{T,Indexer}}{R_{Share}}.
\end{aligned}
\tag{5.11}
$$

Raising $`R_{Host}`$ shrinks the Buffer cost as $`1/R_{Host}`$ but cannot reduce the Indexer cost of a given logical capacity. Moreover, at $`R_{Host}=R^{\mathrm{50pct}}_{Host}`$ the capacity reaches exactly half of the theoretical ceiling; beyond that point the marginal return of raising $`R_{Host}`$ falls off quickly.

#### 5.5 Maximum batch size limits ratio reachability

The worker capacity curve in Section 5.4 contains no $`N_{bs,max}`$, so increasing the maximum batch size does not multiply the model-length lower bound. Concurrency instead enters through the sparse floor: every active sequence must retain at least $`N_{Topk}`$ hot slots. Therefore a given $`R_{Host}`$ can support at most:

$$
N_{bs,max}
\le
\frac{M_{cache}}
{N_{layer}N_{Topk}
\left(
B_{T,MLA}
+\dfrac{R_{Host}B_{T,Indexer}}{R_{Share}}
\right)}
\tag{5.12}
$$

Equivalently, fixing $`N_{bs,max}`$ gives the top-k-imposed ratio ceiling:

$$
R^{Topk}_{Host,max}=\frac{R_{Share}}{B_{T,Indexer}}
\left(
\frac{M_{cache}}
{N_{layer}N_{bs,max}N_{Topk}}
-B_{T,MLA}
\right)
\tag{5.13}
$$

This is why the capacity curve is shared by every batch size while each $`N_{bs,max}`$ has a different colored vertical boundary in the figures. A feasible operating point under this post's default criterion must satisfy both $`C^{GPU}_{Batch}\ge L_{max,model}`$ and $`R_{Host}\le R^{Topk}_{Host,max}`$.

For homogeneous requests of logical length $`L_{req}`$, the same quantities give a useful admission upper bound:

$$
N^{admit}_{req}(R_{Host},L_{req})
\le
\min\left(
N_{bs,max},
\left\lfloor\frac{C^{GPU}_{Batch}(R_{Host})}{L_{req}}\right\rfloor,
\left\lfloor\frac{B^{T,GPU}_{Buffer}(R_{Host})}{N_{Topk}}\right\rfloor
\right)
\tag{5.14}
$$

With a fixed $`6144`$-token Buffer per sequence, $`C_{host,seq}=6144R_{Host}`$ makes the single-request coverage immediately visible. The number of requests still cannot be inferred from $`R_{Host}`$ alone: it also depends on the worker-wide Buffer pool, request lengths, and the top-k floor.

#### 5.6 Engineering takeaways

1. **Check the capacity lower crossing and the top-k ratio ceiling together.** If the lower crossing lies to the right of $`R^{Topk}_{Host,max}`$, that batch size has no feasible $`R_{Host}`$.
2. **In the Buffer-dominated regime, raising $`R_{Host}`$ pays off clearly.** There, a larger host/device ratio buys substantial logical capacity for a small Indexer cost.
3. **In the Indexer-dominated regime, optimize $`R_{Share}`$ first.** $`R_{Share}`$ raises the capacity ceiling linearly, while $`R_{Host}`$ only asymptotically approaches it.
4. **Higher concurrency does not increase worker capacity.** It only divides the fixed Buffer pool more finely and moves the top-k ratio ceiling left.
5. **Do not ship the theoretical bounds as production configs.** Page alignment, allocator metadata, fragmentation, CUDA graphs, and kernel workspaces are all excluded from these formulas. Engineering calculations should apply a safety factor $`0<\eta<1`$, i.e. use $`\eta M_{cache}`$ in place of $`M_{cache}`$.

### 6. Per-Model Evaluation: GLM5.1 and GLM5.2

This section instantiates the model for two concrete NSA models on two H100 deployments. The plotting script is [plot_hisparse_capacity.py](plot_hisparse_capacity.py); every figure's x-axis is $`R_{Host}`$.

| Model | $`N_{layer}`$ | Indexer layout | $`R_{Share}`$ | $`L_{max,model}`$ |
| --- | ---: | --- | ---: | ---: |
| GLM5.1 | $`78`$ | per-layer Indexer cache | $`1`$ | $`262{,}144`$ (256K) |
| GLM5.2 | $`78`$ | Indexer shared across layers ($`78/21`$) | $`3.714`$ | $`1{,}048{,}576`$ (1M) |

| Config | $`M_{Available}`$ | $`F`$ | $`M_{weights}`$ | $`M_{cache}=M_{Available}F-M_{weights}`$ |
| --- | --- | --- | --- | --- |
| H100 DP16EP16 | $`77.47\ \mathrm{GB}`$ | $`0.88`$ | $`64.52\ \mathrm{GB}`$ | $`3.6536\ \mathrm{GB}`$ |
| H100 DP32EP32 | $`77.47\ \mathrm{GB}`$ | $`0.82`$ | $`43.42\ \mathrm{GB}`$ | $`20.1054\ \mathrm{GB}`$ |

#### 6.1 How to read the figures

Each config gets one two-panel figure. The left panel plots the worker-wide logical capacity permitted by the GPU inequality, $`C^{GPU}_{Batch}(R_{Host})`$; its secondary y-axis shows the maximum request count supported when the fixed per-sequence Buffer is $`2048`$, $`4096`$, or $`6144`$ tokens. The right panel shows the absolute Buffer / Indexer HBM composition in GB, with a secondary y-axis for the equivalent pinned host MLA memory. Both x-axes use base-2 logarithmic ticks starting at $`2^{-1}`$.

- Solid curve: worker GPU capacity $`C^{GPU}_{Batch}`$ at the model's $`R_{Share}`$.
- Marked teal / coral / purple curves: the continuous request-count ceilings $`B^{T,GPU}_{Buffer}/2048`$, $`B^{T,GPU}_{Buffer}/4096`$, and $`B^{T,GPU}_{Buffer}/6144`$.
- Markers are placed at the theoretical $`N_{bs}`$ top-k ratio boundaries rather than at arbitrary samples. On the Buffer=$`2048=N_{Topk}`$ curve, each marker therefore lands exactly at the corresponding $`N_{bs}`$ value; the $`4096`$ and $`6144`$ curves land at $`N_{bs}/2`$ and $`N_{bs}/3`$.
- Gray dashed horizontal line: the model-context lower bound $`L_{max,model}`$.
- Blue / gray areas in the memory panel: the absolute Buffer / Indexer HBM usage in GB; their sum is $`M_{cache}`$.
- Marked dark-teal curve in the memory panel: the pinned HostMemory corresponding to $`C^{GPU}_{Batch}`$.

The top-k upper bound is:

$$
R^{Topk}_{Host,max}=\frac{R_{Share}}{B_{T,Indexer}}
\left(
\frac{M_{cache}}
{N_{layer}N_{bs,max}N_{Topk}}
-B_{T,MLA}
\right)
\tag{6.1}
$$

#### 6.2 GLM5.1 (256K, per-layer Indexer)

![GLM5.1 H100 DP16EP16 worker capacity](../assets/glm51_h100_dp16_ep_16.png)

> **Figure conclusion.** Raising $`R_{Host}`$ increases aggregate worker capacity toward the Indexer ceiling, but reduces the maximum request count available to any fixed per-sequence Buffer. A larger Buffer improves each request's hot window at the cost of lower concurrency.

On **DP16EP16** ($`M_{cache}\approx3.65\ \mathrm{GB}`$), the worker reaches the 256K aggregate-capacity target at $`R_{Host}\approx14.05`$. Larger batch sizes do not move this lower crossing, but they move the top-k ceiling left:

| $`N_{bs,max}`$ | Feasible $`R_{Host}`$ for $`C_{Batch}\ge256\mathrm{K}`$ |
| ---: | --- |
| $`1`$ | $`[14.05,\ 128]`$ |
| $`2`$ | $`[14.05,\ 81.67]`$ |
| $`4`$ | $`[14.05,\ 38.35]`$ |
| $`8`$ | $`[14.05,\ 16.69]`$ |

Thus $`N_{bs,max}=8`$ remains reachable, but only in a narrow ratio interval. This does not mean eight simultaneous 256K requests fit; it means the worker has at least one 256K of aggregate logical capacity while retaining the top-k minimum for eight active sequences.

![GLM5.1 H100 DP32EP32 worker capacity](../assets/glm51_h100_dp32_ep_32.png)

> **Figure conclusion.** Raising $`R_{Host}`$ increases aggregate worker capacity toward the Indexer ceiling, but reduces the maximum request count available to any fixed per-sequence Buffer. A larger Buffer improves each request's hot window at the cost of lower concurrency.

On **DP32EP32** ($`M_{cache}\approx20.11\ \mathrm{GB}`$), the larger cache moves the 256K lower crossing down to $`R_{Host}\approx0.77`$ and keeps all four plotted batch sizes reachable:

| $`N_{bs,max}`$ | Feasible $`R_{Host}`$ for $`C_{Batch}\ge256\mathrm{K}`$ |
| ---: | --- |
| $`1`$ | $`[0.77,\ 128]`$ |
| $`2`$ | $`[0.77,\ 256]`$ |
| $`4`$ | $`[0.77,\ 233.4]`$ |
| $`8`$ | $`[0.77,\ 114.2]`$ |

The lower endpoint is shared because worker capacity is independent of how the total Buffer is partitioned. The upper endpoints differ because the total top-k demand grows with $`N_{bs,max}`$.

#### 6.3 GLM5.2 (1M, shared Indexer)

![GLM5.2 H100 DP16EP16 worker capacity](../assets/glm52_h100_dp16_ep_16.png)

> **Figure conclusion.** Raising $`R_{Host}`$ increases aggregate worker capacity toward the Indexer ceiling, but reduces the maximum request count available to any fixed per-sequence Buffer. A larger Buffer improves each request's hot window at the cost of lower concurrency.

On **DP16EP16**, the worker reaches the 1M aggregate-capacity target at $`R_{Host}\approx71.85`$. The feasible intervals are:

| $`N_{bs,max}`$ | Feasible $`R_{Host}`$ for $`C_{Batch}\ge1\mathrm{M}`$ |
| ---: | --- |
| $`1`$ | $`[71.85,\ 512]`$ |
| $`2`$ | $`[71.85,\ 303.3]`$ |
| $`4`$ | $`[71.85,\ 142.4]`$ |
| $`8`$ | empty |

Compared with GLM5.1, the longer context moves the shared lower crossing much farther right. At $`N_{bs,max}=8`$, the top-k ceiling falls below that crossing, so no ratio satisfies both constraints.

![GLM5.2 H100 DP32EP32 worker capacity](../assets/glm52_h100_dp32_ep_32.png)

> **Figure conclusion.** Raising $`R_{Host}`$ increases aggregate worker capacity toward the Indexer ceiling, but reduces the maximum request count available to any fixed per-sequence Buffer. A larger Buffer improves each request's hot window at the cost of lower concurrency.

On **DP32EP32**, the 1M lower crossing falls to $`R_{Host}\approx3.12`$, and all four plotted batch sizes are reachable:

| $`N_{bs,max}`$ | Feasible $`R_{Host}`$ for $`C_{Batch}\ge1\mathrm{M}`$ |
| ---: | --- |
| $`1`$ | $`[3.12,\ 512]`$ |
| $`2`$ | $`[3.12,\ 1024]`$ |
| $`4`$ | $`[3.12,\ 866.9]`$ |
| $`8`$ | $`[3.12,\ 424.2]`$ |

Both models tell the same story: $`N_{bs,max}`$ does not change the worker-capacity curve, but it can make a target unreachable by pushing the top-k ratio ceiling below the model-length crossing. The larger-cache deployment leaves substantially more room between these two boundaries.

### 7. Worked Example: GLM5.2 on H100 DP16EP16

Take $`M_{cache}=77.47\times0.88-64.52\approx3.6536\ \mathrm{GB}`$, $`N_{layer}=78`$, $`N_{bs,max}=8`$, $`R_{Host}=2`$, $`R_{Share}=3.714`$, $`N_{Topk}=2048`$, and the default $`N_{T,Buffer}=6144`$, then walk the three constraints in order.

1. **Sparse floor (4.1):** $`N_{T,Buffer}=6144\ge N_{Topk}=2048`$ — satisfied with $`3\times`$ headroom.
2. **GPU ceiling (4.2):** the per-slot cost is $`B_{T,MLA}+R_{Host}B_{T,Indexer}/R_{Share} = 656+2\times132/3.714\approx727\ \mathrm{B}`$, so the ceiling is $`3.6536\times10^9/(727\times78\times8)\approx8053\ge6144`$ — the default buffer fits. Without Indexer layer sharing ($`R_{Share}=1`$, the GLM5.1 layout) the per-slot cost rises to $`920\ \mathrm{B}`$ and the ceiling drops to $`\approx6364`$: the default buffer would only *barely* fit.
3. **Batch capacity (Lemma 1):** $`C_{host,seq}=2\times6144=12{,}288`$ tokens and $`C_{Batch}=8\times12{,}288=98{,}304`$ tokens, below the 1M aggregate target. The capacity curve reaches 1M at $`R_{Host}\approx71.85`$, but the $`N_{bs,max}=8`$ top-k ceiling is only about $`62`$, so no ratio satisfies both constraints. Reducing $`N_{bs,max}`$ to $`4`$ opens the interval $`[71.85,142.4]`$; alternatively, a larger $`M_{cache}`$ such as DP32EP32 moves the top-k ceiling right.

Conclusion: this configuration runs sparse decode comfortably at $`R_{Host}=2`$, but its worker capacity is only $`98{,}304`$ tokens. Reaching the 1M aggregate target requires simultaneously lowering concurrency (or enlarging $`M_{cache}`$), raising $`R_{Host}`$, and provisioning the host RAM to back the resulting logical capacity.

### 8. Tuning Guide

| Goal | Recommendation |
| --- | --- |
| Run the sparse kernel at all | $`N_{T,Buffer}\ge N_{Topk}`$ |
| Short-sequence fast path | $`N_{T,Buffer}`$ slightly above $`N_{Topk}`$ |
| One model-length of aggregate worker capacity | Raise $`R_{Host}`$ until $`C_{Batch}\ge L_{max,model}`$; verify that the crossing remains below $`R^{Topk}_{Host,max}`$ |
| High concurrency | Use the worker-wide Buffer pool efficiently and verify $`B^{T,GPU}_{Buffer}\ge N_{bs,max}N_{Topk}`$ |
| Host machine RAM | $`\approx1\ \mathrm{TB}`$ supports $`R_{Host}\approx5`$; $`\approx2\ \mathrm{TB}`$ supports $`R_{Host}\approx10`$ |

Trade-offs: enlarging $`N_{T,Buffer}`$ reduces swap-in misses but squeezes the concurrency the GPU can sustain; enlarging $`R_{Host}`$ grows $`C_{host,seq}`$ but inflates the Indexer term in the GPU inequality; a larger $`R_{Share}`$ (more Indexer layer sharing, a model-architecture property) relieves GPU Indexer pressure and raises the capacity ceiling linearly.
