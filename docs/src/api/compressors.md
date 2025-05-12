# Compressors 
```@contents
Pages = ["compressors.md"]
```

## Abstract Types
```@docs
Compressor

CompressorRecipe

CompressorAdjoint

Cardinality

Left

Right
```

## Compressor Structures
```@docs
Gaussian

GaussianRecipe
SparseSign

SparseSignRecipe

```

## Exported  Functions
```@docs
complete_compressor

update_compressor!
```

## Internal Functions
```@docs
RLinearAlgebra.left_mul_dimcheck

RLinearAlgebra.right_mul_dimcheck

RLinearAlgebra.sparse_idx_update!
```
