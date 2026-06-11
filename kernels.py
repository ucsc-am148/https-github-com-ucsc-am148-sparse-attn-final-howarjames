"""STUDENT FILE: implement the three block-sparse rung functions.

Implement these three functions from the spec in ALGORITHMS.md -- no reference
code is shipped:

  dsd_matmul             (A1) block-sparse (BCSR) A @ dense B -> dense C
  sparse_flash_forward   (A2) block-sparse flash attention forward
  sparse_flash_backward  (A3) block-sparse flash attention backward

Your functions must match the signatures below: the SHAPES and DTYPES of the
inputs and outputs (each docstring states them; ALGORITHMS.md sec 0.1 collects
them). EVERYTHING ELSE IS YOURS -- how many @triton.jit kernels you write, the
grid, the (B, H) flatten, strides, output allocation, and the launch/tuning. The
grader asserts the returned shapes and dtypes, then checks correctness against an
fp64 reference.

ALGORITHMS.md is the complete spec: the BCSR layout and its two transpose views,
what each output equals, and the five backward equations.

When `python sanity_check.py` passes all three rungs, you're done.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _dsd_matmul_kernel(values, row_offsets, column_indices, B, C,
                       M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                       BLOCK: tl.constexpr,
                       BLOCK_M: tl.constexpr,
                       BLOCK_N: tl.constexpr):

    # One sparse block row is split into smaller row tiles.
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)

    row_tiles_per_block = BLOCK // BLOCK_M

    # Which sparse block row of A/C are we in?
    row_block = row_tile // row_tiles_per_block

    # Which small row slice inside that sparse block?
    row_subtile = row_tile % row_tiles_per_block

    # Local offsets
    local_m = row_subtile * BLOCK_M + tl.arange(0, BLOCK_M)
    local_k = tl.arange(0, BLOCK)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)

    # Actual dense C rows
    rows = row_block * BLOCK + local_m

    # Accumulator for this small C tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

    # BCSR range for this sparse block row
    start = tl.load(row_offsets + row_block)
    end = tl.load(row_offsets + row_block + 1)

    # Loop over live sparse blocks in this row
    for p in range(start, end):
        k_block = tl.load(column_indices + p)

        # Load A[p, local_m, local_k]
        a = tl.load(
            values
            + p * BLOCK * BLOCK
            + local_m[:, None] * BLOCK
            + local_k[None, :],
            mask=local_m[:, None] < BLOCK,
            other=0.0
        )

        # Load B[k_block * BLOCK + local_k, cols]
        b_rows = k_block * BLOCK + local_k

        b = tl.load(
            B + b_rows[:, None] * N + cols[None, :],
            mask=(b_rows[:, None] < K) & (cols[None, :] < N),
            other=0.0
        )

        acc += tl.dot(a, b, input_precision="ieee")

    # Store C[rows, cols]
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N)
    )

def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    """A1 -- block-sparse C = A @ B. See ALGORITHMS.md sec 1-2.

    Inputs:
      values         (nnz, block, block)  fp32   A's live blocks, row-major
      row_offsets    (M//block + 1,)      int32  per block-row prefix sum of nnz
      column_indices (nnz,)               int32  K-block of each live block
      B              (K, N)               fp32   dense right operand
      M, K, N, block                      ints   dims and block size
    Returns:
      C              (M, N)               fp32

    fp32 throughout, allow_tf32=False.

    TODO: implement.
    """
    
    # Allocate the dense output before passing it into the Triton kernel.
    C = torch.empty((M, N), device=B.device, dtype=torch.float32)

    # Compute only a small row-slice of each sparse block at a time.
    BLOCK_M = 16
    BLOCK_N = 16

    # Each BCSR row-block of height `block` is split into smaller row tiles.
    row_tiles_per_block = block // BLOCK_M

    # Grid axes:
    #   axis 0: small row tiles across all sparse block rows
    #   axis 1: column tiles of the dense output C
    grid = (
        (M // block) * row_tiles_per_block,
        triton.cdiv(N, BLOCK_N),
    )

    # Launch the A1 sparse-dense matmul kernel.
    _dsd_matmul_kernel[grid](
        values, row_offsets, column_indices, B, C,
        M, K, N,
        block, BLOCK_M, BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    return C

@triton.jit
def _sparse_flash_forward_kernel(Q, K, V, O, L, q_row_offsets, q_col_indices,
                                 T: tl.constexpr, D: tl.constexpr,
                                 sm_scale: tl.constexpr,
                                 BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                                 BLOCK_D: tl.constexpr):
    #Precomputed log2e
    log2e = 1.4426950408889634

    # Collapse batch and head into one dimension: bh = b * H + h.
    # The second grid axis chooses the query block.
    bh = tl.program_id(0)
    q_block = tl.program_id(1)

    # Row/column offsets for this query block and the feature dimension.
    q_rows = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_rows = tl.arange(0, BLOCK_K)
    d_cols = tl.arange(0, BLOCK_D)

    # Load the Q block once. It is reused against every live K/V block
    # listed in q_row_offsets/q_col_indices.
    q = tl.load(Q + bh * T * D + q_rows[:, None] * D + d_cols[None, :],
                mask=(q_rows[:, None] < T) & (d_cols[None, :] < D), other=0.0)

    # Online softmax state:
    #   m   = running max score per query row
    #   l   = running denominator sum after max-shifting
    #   acc = running numerator sum, eventually divided by l
    m = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_Q,), tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

    # BCSR query view: this query block attends only the key blocks
    # in q_col_indices[start:end].
    start = tl.load(q_row_offsets + q_block)
    end = tl.load(q_row_offsets + q_block + 1)

    for pidx in range(start, end):
        # Compressed index pidx tells us which real key block to load.
        k_block = tl.load(q_col_indices + pidx)
        offs_k = k_block * BLOCK_K + k_rows

        k = tl.load(K + bh * T * D + offs_k[:, None] * D + d_cols[None, :],
                    mask=(offs_k[:, None] < T) & (d_cols[None, :] < D), other=0.0)
        v = tl.load(V + bh * T * D + offs_k[:, None] * D + d_cols[None, :],
                    mask=(offs_k[:, None] < T) & (d_cols[None, :] < D), other=0.0)

        # Compute QK^T scores. Because we use exp2 below, multiply by log2(e)
        # so exp2(score * log2(e)) equals exp(score).
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * (sm_scale * log2e)

        # Mask rows/columns past T. The sparse pattern already enforces which
        # blocks are allowed; this handles boundary tiles.
        scores = tl.where((q_rows[:, None] < T) & (offs_k[None, :] < T), scores, -float("inf"))

        # Flash/online softmax update. This lets us process one K/V block at a
        # time without ever materializing the full T x T attention matrix.
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l * alpha + tl.sum(p, axis=1)

        # p is computed in fp32, but V is fp16. Triton dot wants matching input
        # dtypes, so cast p to fp16 while accumulating into fp32.
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v, input_precision="ieee")
        m = m_new
        l = l_new

    # Normalize the accumulated numerator and save L = log2(softmax denominator).
    out = acc / l[:, None]
    l_out = m + tl.log2(l)

    tl.store(O + bh * T * D + q_rows[:, None] * D + d_cols[None, :], out,
             mask=(q_rows[:, None] < T) & (d_cols[None, :] < D))
    tl.store(L + bh * T + q_rows, l_out, mask=q_rows < T)

def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    """A2 -- block-sparse flash attention forward. See ALGORITHMS.md sec 1, 3.

    Inputs:
      Q, K, V        (B, H, T, d)         fp16
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query
      q_col_indices  (nnz,)               int32  block i, its live key blocks j
      sm_scale       float                       1/sqrt(d)
      BLOCK_Q, BLOCK_K  ints                     == block (the mask granularity)
    Returns:
      O              (B, H, T, d)         fp16
      L              (B, H, T)            fp32   log2 of the softmax denominator (sec 3)

    See ALGORITHMS.md sec 3 for O and L.

    TODO: implement.
    """
    Bsz, H, T, D = Q.shape
    O = torch.empty_like(Q)
    L = torch.empty((Bsz, H, T), device=Q.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(D)
    grid = (Bsz * H, triton.cdiv(T, BLOCK_Q))
    _sparse_flash_forward_kernel[grid](Q, K, V, O, L, q_row_offsets, q_col_indices,
                                       T, D, sm_scale, BLOCK_Q, BLOCK_K, block_d,
                                       num_warps=4)
    return O, L


@triton.jit
def _compute_D_kernel(O, dO, Dbuf,
                      T: tl.constexpr, D: tl.constexpr,
                      BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr):
    # One program computes delta for one (batch, head, query block).
    bh = tl.program_id(0)
    q_block = tl.program_id(1)
    q_rows = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_cols = tl.arange(0, BLOCK_D)

    o = tl.load(O + bh * T * D + q_rows[:, None] * D + d_cols[None, :],
                mask=(q_rows[:, None] < T) & (d_cols[None, :] < D), other=0.0).to(tl.float32)
    do = tl.load(dO + bh * T * D + q_rows[:, None] * D + d_cols[None, :],
                 mask=(q_rows[:, None] < T) & (d_cols[None, :] < D), other=0.0).to(tl.float32)
    # Backward formula helper:
    #   delta_i = sum_d O[i, d] * dO[i, d]
    delta = tl.sum(o * do, axis=1)
    tl.store(Dbuf + bh * T + q_rows, delta, mask=q_rows < T)


@triton.jit
def _sparse_flash_dq_kernel(Q, K, V, L, dO, Dbuf, dQ, q_row_offsets, q_col_indices,
                            T: tl.constexpr, D: tl.constexpr,
                            sm_scale: tl.constexpr,
                            BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                            BLOCK_D: tl.constexpr):
    # Local constant for exp2-based softmax math.
    log2e = 1.4426950408889634

    # One program computes dQ for one query block in one flattened (B, H).
    bh = tl.program_id(0)
    q_block = tl.program_id(1)

    q_rows = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_rows = tl.arange(0, BLOCK_K)
    d_cols = tl.arange(0, BLOCK_D)

    q = tl.load(Q + bh * T * D + q_rows[:, None] * D + d_cols[None, :],
                mask=(q_rows[:, None] < T) & (d_cols[None, :] < D), other=0.0)
    do = tl.load(dO + bh * T * D + q_rows[:, None] * D + d_cols[None, :],
                 mask=(q_rows[:, None] < T) & (d_cols[None, :] < D), other=0.0)
    lse = tl.load(L + bh * T + q_rows, mask=q_rows < T, other=0.0)
    delta = tl.load(Dbuf + bh * T + q_rows, mask=q_rows < T, other=0.0)

    # Accumulate dQ over every key block attended by this query block.
    acc_dq = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)

    start = tl.load(q_row_offsets + q_block)
    end = tl.load(q_row_offsets + q_block + 1)

    for pidx in range(start, end):
        k_block = tl.load(q_col_indices + pidx)
        offs_k = k_block * BLOCK_K + k_rows

        k = tl.load(K + bh * T * D + offs_k[:, None] * D + d_cols[None, :],
                    mask=(offs_k[:, None] < T) & (d_cols[None, :] < D), other=0.0)
        v = tl.load(V + bh * T * D + offs_k[:, None] * D + d_cols[None, :],
                    mask=(offs_k[:, None] < T) & (d_cols[None, :] < D), other=0.0)

        # Recompute the local attention probabilities for this sparse block.
        # L is already log2(sum exp scores), so p = exp2(scores - L).
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * (sm_scale * log2e)
        scores = tl.where((q_rows[:, None] < T) & (offs_k[None, :] < T), scores, -float("inf"))
        p = tl.exp2(scores - lse[:, None])

        # Backward equations:
        #   dP = dO @ V^T
        #   dS = P * (dP - delta)
        #   dQ += dS @ K
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - delta[:, None])
        acc_dq += tl.dot(ds.to(tl.float16), k, input_precision="ieee")

    # Scores included sm_scale in forward, so gradients through scores also
    # need the sm_scale factor.
    acc_dq *= sm_scale
    tl.store(dQ + bh * T * D + q_rows[:, None] * D + d_cols[None, :], acc_dq,
             mask=(q_rows[:, None] < T) & (d_cols[None, :] < D))


@triton.jit
def _sparse_flash_dkdv_kernel(Q, K, V, L, dO, Dbuf, dK, dV,
                              k_row_offsets, k_col_indices,
                              T: tl.constexpr, D: tl.constexpr,
                              sm_scale: tl.constexpr,
                              BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
                              BLOCK_D: tl.constexpr):
    # This kernel uses the transposed BCSR/key view. One program computes
    # dK and dV for one key block in one flattened (B, H).
    log2e = 1.4426950408889634
    bh = tl.program_id(0)
    k_block = tl.program_id(1)

    q_rows_local = tl.arange(0, BLOCK_Q)
    k_rows = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    d_cols = tl.arange(0, BLOCK_D)

    k = tl.load(K + bh * T * D + k_rows[:, None] * D + d_cols[None, :],
                mask=(k_rows[:, None] < T) & (d_cols[None, :] < D), other=0.0)
    v = tl.load(V + bh * T * D + k_rows[:, None] * D + d_cols[None, :],
                mask=(k_rows[:, None] < T) & (d_cols[None, :] < D), other=0.0)

    # Accumulate contributions from all query blocks that attend this key block.
    acc_dk = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)
    acc_dv = tl.zeros((BLOCK_K, BLOCK_D), tl.float32)

    # Transposed sparse view: for key block j, list all query blocks i
    # such that query block i attends key block j.
    start = tl.load(k_row_offsets + k_block)
    end = tl.load(k_row_offsets + k_block + 1)

    for pidx in range(start, end):
        q_block = tl.load(k_col_indices + pidx)
        q_rows = q_block * BLOCK_Q + q_rows_local

        q = tl.load(Q + bh * T * D + q_rows[:, None] * D + d_cols[None, :],
                    mask=(q_rows[:, None] < T) & (d_cols[None, :] < D), other=0.0)
        do = tl.load(dO + bh * T * D + q_rows[:, None] * D + d_cols[None, :],
                     mask=(q_rows[:, None] < T) & (d_cols[None, :] < D), other=0.0)
        lse = tl.load(L + bh * T + q_rows, mask=q_rows < T, other=0.0)
        delta = tl.load(Dbuf + bh * T + q_rows, mask=q_rows < T, other=0.0)

        # Recompute attention probabilities for this (query block, key block).
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * (sm_scale * log2e)
        scores = tl.where((q_rows[:, None] < T) & (k_rows[None, :] < T), scores, -float("inf"))
        p = tl.exp2(scores - lse[:, None])

        # dV accumulates P^T @ dO.
        acc_dv += tl.dot(tl.trans(p).to(tl.float16), do, input_precision="ieee")

        # dK accumulates dS^T @ Q, where dS = P * (dP - delta).
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - delta[:, None])
        acc_dk += tl.dot(tl.trans(ds).to(tl.float16), q, input_precision="ieee")

    # Gradient through S = sm_scale * QK^T.
    acc_dk *= sm_scale
    tl.store(dK + bh * T * D + k_rows[:, None] * D + d_cols[None, :], acc_dk,
             mask=(k_rows[:, None] < T) & (d_cols[None, :] < D))
    tl.store(dV + bh * T * D + k_rows[:, None] * D + d_cols[None, :], acc_dv,
             mask=(k_rows[:, None] < T) & (d_cols[None, :] < D))


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,   # key-block view (sec 1)
                          q_row_offsets, q_col_indices,   # query-block view (sec 1)
                          sm_scale, BLOCK_Q, BLOCK_K):
    """A3 -- block-sparse flash attention backward. See ALGORITHMS.md sec 1, 4.

    Inputs:
      Q, K, V, O, dO (B, H, T, d)         fp16   O, dO are the forward output and its grad
      L              (B, H, T)            fp32   the forward residual
      k_row_offsets  (T//block + 1,)      int32  key-block view: for key block j,
      k_col_indices  (nnz,)               int32  the query blocks i that attend it
      q_row_offsets  (T//block + 1,)      int32  query-block view: for query block i,
      q_col_indices  (nnz,)               int32  its key blocks j (same as forward)
      sm_scale       float
      BLOCK_Q, BLOCK_K  ints                     == block
    Returns:
      dQ, dK, dV     (B, H, T, d)         fp16

    See ALGORITHMS.md sec 4 for the five gradient equations.

    TODO: implement.
    """
    Bsz, H, T, D = Q.shape

    # Allocate gradient outputs with the required fp16 shapes.
    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)

    # Temporary fp32 buffer for delta = rowsum(O * dO).
    Dbuf = torch.empty((Bsz, H, T), device=Q.device, dtype=torch.float32)

    # Triton dot dimensions usually need powers of two.
    block_d = triton.next_power_of_2(D)

    # Flatten batch and head for kernel launches.
    bh = Bsz * H
    q_grid = (bh, triton.cdiv(T, BLOCK_Q))
    k_grid = (bh, triton.cdiv(T, BLOCK_K))

    # Step 1: compute delta helper used by all backward equations.
    _compute_D_kernel[q_grid](O, dO, Dbuf, T, D, BLOCK_Q, block_d, num_warps=4)
    # Step 2: compute dQ using the normal query-block sparse view.
    _sparse_flash_dq_kernel[q_grid](Q, K, V, L, dO, Dbuf, dQ, q_row_offsets, q_col_indices,
                                    T, D, sm_scale, BLOCK_Q, BLOCK_K, block_d,
                                    num_warps=4)

    # Step 3: compute dK and dV using the transposed/key-block sparse view.
    _sparse_flash_dkdv_kernel[k_grid](Q, K, V, L, dO, Dbuf, dK, dV,
                                      k_row_offsets, k_col_indices,
                                      T, D, sm_scale, BLOCK_Q, BLOCK_K, block_d,
                                      num_warps=4)
    return dQ, dK, dV