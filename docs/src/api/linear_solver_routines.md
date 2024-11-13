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
```

## Vector Row Solvers

```@docs
LinSysVecRowProjStd

LinSysVecRowProjPO

LinSysVecRowProjFO

IterativeHessianSketch
```

## Vector Column Solvers

```@docs
LinSysVecColProjStd

LinSysVecColProjPO

LinSysVecColProjFO
```

## Block Row Solver

```@docs
LinSysBlkRowLQ
```

## Block Column Solver

```@docs
LinSysBlkColGent
```

## Solving Routine

```@docs
RLinearAlgebra.rsubsolve!
```

## Alternative Solvers

```@docs
arnoldi!

randomized_arnoldi!
```