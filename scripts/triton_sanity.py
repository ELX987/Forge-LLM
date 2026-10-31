import torch, triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

n = 1_000_003
x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda"); o = torch.empty_like(x)
add_kernel[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
torch.testing.assert_close(o, x + y)
print("triton", triton.__version__, "vector add ok")
