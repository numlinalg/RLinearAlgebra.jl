# Linear Subsolvers

```@contents
Pages=["linear_solver_routines.md"]
```

## Abstract Types

```@docs
LinSysSolveRoutine

LinSysVecRowProjection

LinSysVecColProjection

LinSysBlkRowProjection

LinSysBlkColProjection

LinSysPreconKrylov

LinSysBlkRowProj

LinSysBlkColProj
```

## Vector Row Solvers

```@docs
LinSysVecRowProjStd

LinSysVecRowProjPO

LinSysVecRowProjFO
```

## Vector Column Solvers

```@docs
LinSysVecColProjStd

LinSysVecColProjPO

LinSysVecColProjFO
```

## Block Row Solver

```
LinSysBlkRowProj
```

## Block Column Solver

```
LinSysBlkColProj
```

## Subsolver Function

```@docs
RLinearAlgebra.rsubsolve!(::Nothing,::AbstractVector,::Tuple,::Int64)
```
