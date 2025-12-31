# Solving a Consistent Linear System

This guide demonstrates how to use `RLinearAlgebra.jl` package to solve a 
**consistent linear system**. That is, how to approximately find $x$
that satisfies

$$Ax = b,$$

where $A$ is a matrix; and $b$ is a vector in the column space of $A$.

First, let's define our linear system $Ax = b$ with some known solution `x_true`.

```@example ConsistentExample; continued=true
num_rows, num_cols = 100, 5;
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);
b = A * x_true;
```


`RLinearAlgebra.jl` can find an approximate solution using the generalized 
Kaczmarz method [patel2023randomized](@cite).

```@example ConsistentExample; continued=true
using RLinearAlgebra
solver = Kaczmarz(log = BasicLogger(max_it = 300))
solution = zeros(Float64, num_cols)
rsolve!(solver, solution, A, b)
```
`rsolve!` updates `solution` with the approximate solution.