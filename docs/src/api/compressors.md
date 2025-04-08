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
RLinearAlgebra.left_mat_mul_dimcheck

RLinearAlgebra.right_mat_mul_dimcheck

RLinearAlgebra.vec_mul_dimcheck

RLinearAlgebra.update_row_idxs!
```
