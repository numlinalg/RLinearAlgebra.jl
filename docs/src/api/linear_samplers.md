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

LinSysBlkRowGaussSampler

LinSysBlkRowReplace

LinSysBlkRowRandCyclic

LinSysBlkRowSelectWoReplacement

LinSysBlkRowFJLT

LinSysBlkRowSRHT

LinSysBlkRowSparseSign
```

## Block Vector Col Samplers

```@docs
LinSysBlkColCountSketch

LinSysBlkColGaussSampler

LinSysBlkColReplace

LinSysBlkColRandCyclic

LinSysBlkColSelectWoReplacement

LinSysBlkColFJLT

LinSysBlkColSRHT

LinSysBlkColSparseSign
```

## Sample Function
```@docs
RLinearAlgebra.sample(::Nothing,::AbstractArray,::AbstractVector,
    ::AbstractVector,::Int64)
```

## Internal Functions
```@docs
RLinearAlgebra.init_blocks_cyclic!

RLinearAlgebra.fwht!
```
