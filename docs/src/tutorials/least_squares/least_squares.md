# Solving a Least Squares Problem

This guide demonstrates how to use `RLinearAlgebra.jl` package to find the 
**least-squares solution** to a linear system. This is typically used for 
*inconsistent* systems, where no exact $Ax=b$ solution exists.

The goal is to find the vector $x$ that minimizes the squared Euclidean norm 
of the residual:

$$\min_{x} \|Ax - b\|_2^2$$

We'll walk through a simple example of setting up and solving such a problem.

First, let's define an *inconsistent* linear system. We'll do this by creating a 
known `x_true` and adding noise to make $b \approx Ax_{\text{true}}$.

```julia
using LinearAlgebra
num_rows, num_cols = 1000, 50;
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);
noise = 0.1 * randn(Float64, num_rows);
b = (A * x_true) + noise;
```

```@setup LeastSquaresExample
using LinearAlgebra
num_rows, num_cols = 1000, 50;
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);
noise = 0.1 * randn(Float64, num_rows);
b = (A * x_true) + noise;
```
`RLinearAlgebra.jl` can find the least-squares solution using the same 
simple interface as in the consistent system example:

```@example LeastSquaresExample
using RLinearAlgebra
solver = Kaczmarz(log = BasicLogger(max_it = 300))
solution = zeros(Float64, num_cols)
rsolve!(solver, solution, A, b)
```
`rsolve!` finds the least-squares solution and stores it in the solution vector.
