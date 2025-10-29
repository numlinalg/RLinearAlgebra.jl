# Solving a Consistent Linear System

This guide demonstrates how to use `RLinearAlgebra.jl` package to solve a 
**consistent linear system**---a system where at least one solution 
exists---expressed in the form:

$$Ax = b.$$

We'll walk through setting up the problem, using a solver, and verifying the result.


First, let's define our linear system $Ax = b$ with some known solution `x_true`.


```julia
using LinearAlgebra
num_rows, num_cols = 1000, 50;
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);
b = A * x_true;
```

```@setup ConsistentExample
using LinearAlgebra
num_rows, num_cols = 1000, 50;
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);
b = A * x_true;
```
`RLinearAlgebra.jl` can solve this system in just a few lines of codes:

```@example ConsistentExample
using RLinearAlgebra
solver = Kaczmarz(log = BasicLogger(max_it = 300))
solution = zeros(Float64, num_cols)
rsolve!(solver, solution, A, b)
```
`rsolve!` puts the solution in the vector `solution`.












