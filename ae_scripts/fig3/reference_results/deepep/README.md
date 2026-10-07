# DeepEP reference-result provenance

The four CSV files in this directory are the released Fig. 3 plot inputs.
They record token counts and measured dispatch/combine timings, but do not
record `num_experts` or the full benchmark command. The original launcher that
produced them is not present in the surviving paper artifact.

Consequently, these CSVs cannot establish whether their source run used the
DeepSeek-V3 value of 256 routed experts or the generic DeepEP
`test_low_latency.py` default of 288. The latter default is not a DeepSeek-V3
model parameter and must not be presented as such.

Fresh AE measurement uses 256 routed experts, matching
`DeepSeek-V3/config.json` (`n_routed_experts=256`). Plot-only reproduction
retains these released numbers without making an unsupported expert-count
claim.
