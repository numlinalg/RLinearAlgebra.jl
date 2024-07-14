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

## Vector Column Samplers

```@docs
LinSysVecColDetermCyclic

LinSysVecColOneRandCyclic
```

## Block Vector Row Samplers

```@docs
LinSysBlkRowCountSketch

LinSysBlkColSelectWoReplacement

LinSysBlkRowSelectWoReplacement
```

## Sample Function
```@docs
RLinearAlgebra.sample(::Nothing,::AbstractArray,::AbstractVector,
    ::AbstractVector,::Int64)
```
