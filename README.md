# Reproducing ROCm error on AMD AAC
<img width="1468" alt="image" src="https://github.com/user-attachments/assets/88a43205-7124-4c7b-9012-7524b9dde927" />
The "block dimensions: 256x1x1: hipError_t(9)" is universal; the specific operation (e.g., "fp32_comparison", "bf16_comparison") can be different depending on the specific implementation.

## Steps
### Environment
```bash
python3 -m venv hf
source hf/bin/activate
python -m pip install jax[rocm] jax-rocm60-pjrt jax-rocm60-plugin flax

git clone --recursive https://github.com/jingyang-kylist/rocm_error_reproduce.git
cd rocm_error_reproduce/transformers
python -m pip install .
cd ..
```

### Running
```bash
salloc --reservation=t20-03_reservation --exclusive --mem=0
module load rocm-6.3.0
source hf/bin/activate

# under rocm_error_reproduce/
# the default args will trigger the error
# max_batch_size = 1, max_seq_len = 16384
python main.py
# shorter seq len can work, e.g.
# python main.py --max_seq_len 32
# some other configurations can also trigger the error
```
