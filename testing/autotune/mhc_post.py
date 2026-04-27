import argparse

import tilelang
import tilelang.language as T
import torch
import os

tilelang.cache.clear_cache()

os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

parser = argparse.ArgumentParser(description="MHC Post CV Kernel Autotune")
parser.add_argument("--batch", type=int, default=4, help="Batch dimension")
parser.add_argument("--m", type=int, default=1024, help="M dimension")
parser.add_argument("--n", type=int, default=512, help="N dimension (hidden)")
parser.add_argument("--r", type=int, default=64, help="R dimension (residual intermediate)")
args = parser.parse_args()

BATCH = args.batch
M = args.m
N = args.n
R = args.r

def ref_prog(X, post_layer_mix, comb_res_mix, residual):
    term1 = X * post_layer_mix.unsqueeze(0).unsqueeze(0)
    term2 = comb_res_mix.transpose(-1, -2) @ residual.unsqueeze(-1)
    return term1 + term2.squeeze(-1).unsqueeze(1)

def get_config():
    return [
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_R": 16},
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_R": 16},
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_R": 32},
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_R": 32},
    ]

def supply_prog(params):
    torch.manual_seed(0)
    return [
        torch.randn(BATCH, M, N).half().npu(),
        torch.randn(BATCH, N).half().npu(),
        torch.randn(BATCH, R, N).half().npu(),
        torch.randn(BATCH, R).half().npu(),
    ]

@tilelang.autotune(
    configs=get_config(),
    ref_prog=ref_prog,
    supply_prog=supply_prog,
    atol=1e-2,
    rtol=1e-2,
)
@tilelang.jit(out_idx=[-1], target="npuir")
def mhc_post(
    BATCH: int,
    M: int,
    N: int,
    R: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_R: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @T.prim_func
    def main(
        X: T.Tensor((BATCH, M, N), dtype),
        post_layer_mix: T.Tensor((BATCH, N), dtype),
        comb_res_mix: T.Tensor((BATCH, R, N), dtype),
        residual: T.Tensor((BATCH, R), dtype),
        Out: T.Tensor((BATCH, M, N), dtype),
    ):
        with T.Kernel(BATCH * T.ceildiv(M, BLOCK_M), is_npu=True) as (cid, _):
            batch_id = cid // T.ceildiv(M, BLOCK_M)
            bm = cid % T.ceildiv(M, BLOCK_M)
            
            X_shared = T.alloc_shared((BLOCK_M, BLOCK_N), dtype)
            post_local = T.alloc_shared((BLOCK_N), dtype)
            
            gemv_result = T.alloc_fragment((1, BLOCK_N), accum_dtype)
            gemv_broadcast = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            out_accum = T.alloc_fragment((BLOCK_M, BLOCK_N), accum_dtype)
            
            num_N_steps = T.ceildiv(N, BLOCK_N)
            num_R_steps = T.ceildiv(R, BLOCK_R)
            
            for n_step in T.serial(num_N_steps):
                n_start = n_step * BLOCK_N
                n_end = T.min(n_start + BLOCK_N, N)
                curr_BLOCK_N = n_end - n_start
                
                T.clear(gemv_result)
                
                for r_step in T.serial(num_R_steps):
                    r_start = r_step * BLOCK_R
                    r_end = T.min(r_start + BLOCK_R, R)
                    curr_BLOCK_R = r_end - r_start
                    
                    residual_row = T.alloc_shared((1, curr_BLOCK_R), dtype)
                    comb_slice = T.alloc_shared((curr_BLOCK_R, curr_BLOCK_N), dtype)
                    
                    for r_i in T.Parallel(curr_BLOCK_R):
                        residual_row[0, r_i] = residual[batch_id, r_start + r_i]
                    
                    for r_i, n_i in T.Parallel(curr_BLOCK_R, curr_BLOCK_N):
                        comb_slice[r_i, n_i] = comb_res_mix[batch_id, r_start + r_i, n_start + n_i]
                    
                    T.gemm(
                        residual_row,
                        comb_slice,
                        gemv_result,
                        size=[1, curr_BLOCK_R, curr_BLOCK_N],
                        initC=(r_step == 0)
                    )
                
                T.vbrc(gemv_result, gemv_broadcast)
                
                T.copy(X[batch_id, bm * BLOCK_M, n_start], X_shared, size=[BLOCK_M, curr_BLOCK_N])
                T.copy(post_layer_mix[batch_id, n_start], post_local, size=[curr_BLOCK_N])
                
                T.vmul(X_shared, post_local, out_accum)
                T.vadd(out_accum, gemv_broadcast, out_accum)
                
                T.copy(out_accum, Out[batch_id, bm * BLOCK_M, n_start], size=[BLOCK_M, curr_BLOCK_N])

    return main

func = mhc_post(BATCH, M, N, R)

print("Best Config:", func.get_tuner_result())
print("Test passed!")