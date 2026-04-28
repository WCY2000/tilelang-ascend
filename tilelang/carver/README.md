# Carver: A Tile-Structure Based Hint Recommend Framework for Ascend NPU

**Carver** is a lightweight framework for generating and ranking tile configurations (tiling strategies) for Ascend NPU backends. It helps you explore efficient mappings of loops for operations such as matrix multiplication, elementwise transforms, and other reduction-oriented kernels.

Carver combines hardware architecture information, user-defined tile structures, and built-in heuristics to recommend tiling strategies (or "hints").

---

### Key Features
- **Unified Tiling Framework**: Generate tile candidates under a unified API.
- **Architecture-Specific Modeling**: Take into account Ascend NPU constraints (UB size, L1/L0 buffer sizes, CUBE unit specs).
- **Flexible Templates**: High-level templates (like `MatmulTemplate`, `GeneralReductionTemplate`, `ElementwiseTemplate`) let you concisely specify kernel structures.

---

## Usage Examples

### Basic Usage: General Reduction Template

```python
from tilelang import carver
from tilelang.utils.npu_arch import AscendArch

arch = AscendArch()

carve_template = carver.GeneralReductionTemplate(
    structure="SSR",
    shape=[1024, 1024, 1024],
    dtype="float16",
).with_arch(arch)

hints = carve_template.recommend_hints(topk=20)
for hint in hints:
    print(hint)
```

### Matmul Template

```python
from tilelang import carver
from tilelang.utils.npu_arch import AscendArch

arch = AscendArch()
carve_template = carver.MatmulTemplate(
    M=1024,
    N=1024,
    K=1024,
    in_dtype="float16",
    accum_dtype="float16",
    out_dtype="float16",
).with_arch(arch)

func = carve_template.equivalent_function()
print("Equivalent Function:\n", func)

hints = carve_template.recommend_hints(topk=20)
for hint in hints:
    print(hint)
```

---

## Supported Templates

- **`GeneralReductionTemplate`**: For general `Spatial-Spatial-Reduce` (SSR) structures.
- **`FlashAttentionTemplate`**: For attention-like operations.
- **`MatmulTemplate`**: For standard matrix multiplication `C = A * B`.
- **`GEMVTemplate`**: For `y = Ax` style operations.
- **`ElementwiseTemplate`**: For elementwise transformations.