# Linear Solver Logs

```@contents
Pages = ["linear_solver_logs.md"]
```


## Abstract Types

```@docs
LinSysSolverLog
```

## Loggers

```@docs
LSLogOracle

LSLogFull

LSLogFullMA
```

## Log Function

```@docs
RLinearAlgebra.log_update!(::Nothing,::LinSysSampler,::AbstractVector,::Tuple,
  ::Int64,::Any,::Any)
```

## Log Specific Functions

```@docs
RLinearAlgebra.get_uncertainty(::LSLogFullMA; alpha = .05)
```
