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

Undef
```

## Compressor Structures
```@docs
CountSketch

CountSketchRecipe

FJLT

FJLTRecipe

Gaussian

GaussianRecipe

Sampling

SamplingRecipe

SparseSign

SparseSignRecipe

SRHT 

SRHTRecipe
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

RLinearAlgebra.fwht!
```
