"""Verify a CUDA kernel before loading the speech model."""
import torch

if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available to this Python environment.')

device = torch.device('cuda:0')
arch = f'sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}'
if arch not in torch.cuda.get_arch_list():
    raise SystemExit(f'Installed PyTorch does not support {arch}: {torch.cuda.get_arch_list()}')

# This checks allocation, a BF16 matrix-multiply kernel, and synchronization.
# It is deliberately independent of Qwen and browser capture.
left = torch.randn((512, 512), device=device, dtype=torch.bfloat16)
right = torch.randn((512, 512), device=device, dtype=torch.bfloat16)
result = left @ right
torch.cuda.synchronize()
print(f'CUDA smoke test passed: {torch.cuda.get_device_name(0)}; {arch}; checksum={result.float().mean().item():.5f}')
