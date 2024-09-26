# Linear Samplers

```@contents
Pages = ["linear_samplers.md"]
```

## Abstract Types

```@docs
LinSysSampler

LinSysVecRowSampler

LinSysVecColSampler

LinSysBlkRowSampler

LinSysBlkColSampler
```

## Vector Row Samplers

```@docs
LinSysVecRowDetermCyclic

LinSysVecRowHopRandCyclic

LinSysVecRowOneRandCyclic

LinSysVecRowPropToNormSampler

LinSysVecRowRandCyclic

LinSysVecRowUnidSampler

LinSysVecRowUnifSampler

LinSysVecRowSparseUnifSampler

LinSysVecRowGaussSampler

LinSysVecRowSparseGaussSampler

LinSysVecRowMaxResidual

LinSysVecRowResidCyclic

LinSysVecRowMaxDistance

LinSysVecRowDistCyclic
```
## Block Vector Row Samplers

```@docs
LinSysBlockRowFJLT

LinSysBlockRowSRHT
```

## Vector Column Samplers

```@docs
LinSysVecColDetermCyclic

LinSysVecColOneRandCyclic
```

<<<<<<< HEAD
## Block Vector Row Samplers

```@docs
LinSysBlkRowGaussSampler

LinSysBlkRowReplace

LinSysBlkRowRandCyclic
```

## Block Vector Col Samplers

```@docs
LinSysBlkColGaussSampler

LinSysBlkColReplace

LinSysBlkColRandCyclic
=======
## Block Vector Col Samplers

```@docs
LinSysBlockColFJLT

LinSysBlockColSRHT
>>>>>>> 20e6c85 (Added FJLT for column)
```

## Sample Function
```@docs
RLinearAlgebra.sample(::Nothing,::AbstractArray,::AbstractVector,
    ::AbstractVector,::Int64)
```

## Internal Functions
```@docs
RLinearAlgebra.init_blocks_cyclic!
```
