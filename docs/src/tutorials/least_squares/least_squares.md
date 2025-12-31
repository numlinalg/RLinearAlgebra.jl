# Solving a Least Squares Problem

This guide demonstrates how to use `RLinearAlgebra.jl` package to approximately
solve a linear least squares problem,

$$\min_{x} \|Ax - b\|_2^2.$$


First, define $A$ and $b$.

```@example LeastSquaresExample; continued=true
num_rows, num_cols = 1000, 50
A = randn(Float64, num_rows, num_cols)
b = A*randn(Float64, num_cols) + 0.1*randn(Float64, num_rows)
```

`RLinearAlgebra.jl` can find the least-squares solution using the same 
simple interface as in the consistent system example:

```@example LeastSquaresExample; continued=true
using RLinearAlgebra
solver = Kaczmarz(log = BasicLogger(max_it = 300))
solution = zeros(Float64, num_cols) # Initial guess of zeros
rsolve!(solver, solution, A, b)
```
`rsolve!` finds the least-squares solution and stores it in the solution vector.
