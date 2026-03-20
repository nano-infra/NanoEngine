"""Unit test to inspect npu_moe_init_routing behavior.

Run on a single NPU:
    python tests/test_moe_init_routing.py
"""

import torch
import torch_npu


def test_moe_init_routing():
    device = "npu:0"
    torch.npu.set_device(device)

    # Small example: 4 tokens, top_k=2, 4 experts
    T, K, num_experts = 4, 2, 4
    H = 8  # small hidden size
    active_num = T * K  # 8

    # Deterministic hidden states: token i has all values = i+1
    x = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
    for i in range(T):
        x[i] = float(i + 1)

    # Expert assignments (known):
    # Token 0 → experts [1, 3]
    # Token 1 → experts [0, 2]
    # Token 2 → experts [3, 1]
    # Token 3 → experts [2, 0]
    expert_idx = torch.tensor(
        [[1, 3], [0, 2], [3, 1], [2, 0]], dtype=torch.int32, device=device
    )

    print("=" * 60)
    print("INPUT")
    print("=" * 60)
    print(f"x (each row = token value):\n{x.float()}")
    print(f"expert_idx:\n{expert_idx}")
    print(f"T={T}, K={K}, active_num={active_num}")

    # ---------------------------------------------------------------
    # Test 1: row_idx[i, k] = i  (token index)
    # ---------------------------------------------------------------
    row_idx_token = (
        torch.arange(T, device=device).unsqueeze(1).expand(-1, K)
    ).to(torch.int32)

    print("\n" + "=" * 60)
    print("TEST 1: row_idx[i,k] = i (token index)")
    print("=" * 60)
    print(f"row_idx:\n{row_idx_token}")

    ex, er, ee = torch_npu.npu_moe_init_routing(
        x, row_idx_token, expert_idx, active_num
    )

    print(f"\nexpanded_x (first col only, identifies token):\n{ex[:, 0].float()}")
    print(f"expanded_row_idx:\n{er}")
    print(f"expanded_expert_idx:\n{ee}")

    # Verify which token each expanded_x row came from
    token_from_data = []
    for i in range(active_num):
        val = ex[i, 0].float().item()
        token_from_data.append(int(val) - 1 if val > 0 else -1)
    print(f"token source (from data):  {token_from_data}")
    print(f"expanded_row_idx values:   {er.tolist()}")
    print(f"expanded_expert_idx values:{ee.tolist()}")

    # Check: is expanded_row_idx our input row_idx or internal flat index?
    print("\n--- Analysis ---")
    print(f"Max expanded_row_idx = {er.max().item()}")
    print(f"If passthrough (row_idx=i): max should be {T - 1}")
    print(f"If internal i*K+k:         max should be {T * K - 1}")

    for j in range(active_num):
        tok_data = token_from_data[j]
        row_val = er[j].item()
        exp_val = ee[j].item()
        # If expanded_row_idx is our input row_idx (= token index):
        tok_from_rowid = row_val
        # If expanded_row_idx is internal i*K+k:
        tok_from_divmod = row_val // K
        k_from_divmod = row_val % K
        print(
            f"  [{j}] expert={exp_val:2d}  data_token={tok_data:2d}  "
            f"row_idx={row_val:3d}  "
            f"as_token_id={tok_from_rowid:2d}  "
            f"divmod=({tok_from_divmod},{k_from_divmod})"
        )

    # ---------------------------------------------------------------
    # Test 2: row_idx[i, k] = i*K+k  (expanded slot index)
    # ---------------------------------------------------------------
    row_idx_expanded = (
        torch.arange(T, device=device).unsqueeze(1).expand(-1, K) * K
        + torch.arange(K, device=device).unsqueeze(0)
    ).to(torch.int32)

    print("\n" + "=" * 60)
    print("TEST 2: row_idx[i,k] = i*K+k (expanded slot index)")
    print("=" * 60)
    print(f"row_idx:\n{row_idx_expanded}")

    ex2, er2, ee2 = torch_npu.npu_moe_init_routing(
        x, row_idx_expanded, expert_idx, active_num
    )

    print(f"\nexpanded_x (first col):\n{ex2[:, 0].float()}")
    print(f"expanded_row_idx:\n{er2}")
    print(f"expanded_expert_idx:\n{ee2}")

    token_from_data2 = []
    for i in range(active_num):
        val = ex2[i, 0].float().item()
        token_from_data2.append(int(val) - 1 if val > 0 else -1)
    print(f"token source (from data):  {token_from_data2}")

    print("\n--- Comparison ---")
    print(f"Test1 expanded_row_idx: {er.tolist()}")
    print(f"Test2 expanded_row_idx: {er2.tolist()}")
    print(f"Are they the same?     {torch.equal(er, er2)}")

    # ---------------------------------------------------------------
    # Test 3: Verify correct weight lookup with divmod
    # ---------------------------------------------------------------
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.5, 0.5], [0.6, 0.4], [0.8, 0.2]],
        dtype=torch.bfloat16,
        device=device,
    )

    print("\n" + "=" * 60)
    print("TEST 3: Verify weight recovery via divmod (Test 1 row_idx=i)")
    print("=" * 60)
    print(f"topk_weights:\n{topk_weights.float()}")

    abs_er = torch.abs(er).to(torch.int64)
    tok_via_divmod = abs_er // K
    kslot_via_divmod = abs_er % K

    print(f"\nFor each sorted entry:")
    for j in range(active_num):
        tok = tok_via_divmod[j].item()
        ks = kslot_via_divmod[j].item()
        exp = ee[j].item()
        actual_tok = token_from_data[j]
        w = topk_weights[tok, ks].float().item() if tok < T and ks < K else -1
        # Check: does expert_idx[tok, ks] == exp?
        expected_exp = expert_idx[tok, ks].item() if tok < T and ks < K else -1
        match = "OK" if (actual_tok == tok and expected_exp == exp) else "MISMATCH"
        print(
            f"  [{j}] expert={exp:2d}  data_tok={actual_tok}  "
            f"divmod_tok={tok} divmod_k={ks}  "
            f"expected_expert={expected_exp}  weight={w:.2f}  {match}"
        )


if __name__ == "__main__":
    test_moe_init_routing()
