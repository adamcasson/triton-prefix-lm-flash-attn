"""
Modification for prefix LM of original Triton FlashAttention unit test
https://github.com/openai/triton/blob/d69eb70a602aba0ab086d8d22ec6f4316e03c6be/python/test/unit/operators/test_flash_attention.py
"""

import pytest
import torch

from prefix_flash_attention import prefix_attention


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [  #
    (2, 4, 512, 16),
    (2, 4, 512, 32),
    (2, 4, 512, 64),
    (2, 4, 512, 128),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seq_par', [True, False])
def test_op(Z, H, N_CTX, D_HEAD, dtype, seq_par):
    import os
    enable_tma = os.environ.get('ENABLE_TMA', 'not found').lower()
    if enable_tma in ["on", "true", "1"]:
        if dtype == torch.bfloat16:
            pytest.skip('bfloat16 tma not support currently')

    capability = torch.cuda.get_device_capability()
    interpreter = os.environ.get("TRITON_INTERPRET", 'not found') in ["on", "true", "1"]
    if not interpreter and capability[0] < 8:
        pytest.skip("Flash attention only supported for compute capability >= 80")
    torch.manual_seed(20)
    q = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = D_HEAD ** -0.5
    prefix_len = 256
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    M[:prefix_len, :prefix_len] = 1
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # # triton implementation
    tri_out = prefix_attention(q, k, v, prefix_len, sm_scale, seq_par)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    atol = 1e-1 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_out), dim=0),
                               torch.nn.functional.normalize(torch.flatten(tri_out), dim=0), atol=atol, rtol=0)
    torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_dv), dim=0),
                               torch.nn.functional.normalize(torch.flatten(tri_dv), dim=0), atol=atol, rtol=0)
    torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_dk), dim=0),
                               torch.nn.functional.normalize(torch.flatten(tri_dk), dim=0), atol=atol, rtol=0)
    torch.testing.assert_close(torch.nn.functional.normalize(torch.flatten(ref_dq), dim=0),
                               torch.nn.functional.normalize(torch.flatten(tri_dq), dim=0), atol=atol, rtol=0)