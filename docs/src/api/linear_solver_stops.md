# Linear Solver Stop Criteria

```@contents
Pages=["linear_solver_stops.md"]
```

## Abstract Types
```@docs
LinSysStopCriterion
```

## Stopping Criteria
```@docs
LSStopMaxIterations
LSStopThreshold
LSStopMA
```

## Stopping Function

```@docs
RLinearAlgebra.check_stop_criterion(log::LinSysSolverLog, stop::LinSysStopCriterion)

```
