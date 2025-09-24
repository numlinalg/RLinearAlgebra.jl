# A simple example


## Use Case: Solving a Least-Squares Problem with the Sparse Sign Method

This guide demonstrates how to use the `SparseSign` compression method from the `RLinearAlgebra.jl` package to solve an overdetermined linear system (i.e., a least-squares problem) of the form:

$$\min_{x} \|Ax - b\|_2^2$$

We will follow the design philosophy of `RLinearAlgebra.jl` by composing different modules (`Solver`, `Compressor`, `Logger`, etc.) to build and solve the problem.

---
### 1. Problem Setup

Let's define a specific linear system $Ax = b$. 

To verify the accuracy of the final result, suppose that we know the true solution of the system, $x_{\text{true}}$, and then use it and a random generated matrix $A$ to generate the vector $b$.

* **Matrix `A`**: A random $100 \times 20$ matrix.
* **Vector `b`**: Calculated as $b = A x_{\text{true}}$, with dimensions $100 \times 1$.
* **Goal**: Find a solution $x$ that is as close as possible to $x_{\text{true}}$.

To achieve this, we need to import the required libraries and create the matrix `A` and vector `b` as defined above. We will also set an initial guess, `x_init`, for the solver.

```@example SparseSignExample
# Import relevant libraries
using RLinearAlgebra, Random, LinearAlgebra


# Define the dimensions of the linear system
num_rows, num_cols = 100, 20

# Create the matrix A and a known true solution x_true
A = randn(Float64, num_rows, num_cols);
x_true = randn(Float64, num_cols);

# Calculate the right-hand side vector b from A and x_true
b = A * x_true;

# Set an initial guess for the solution vector x (typically a zero vector)
x_init = zeros(Float64, num_cols);

println("Dimensions:")
println(" - Matrix A: ", size(A))
println(" - Vector b: ", size(b))
println(" - True solution x_true: ", size(x_true))
println(" - Initial guess x_init: ", size(x_true))
```

---
### 2. Configure the `SparseSign` Compressor

The idea of randomized methods is to reduce the scale of the original problem when the dimention of matrix $A$ is too big, using a random "sketch" or "compression" matrix, $S$. Here, we choose `SparseSign` as our `Compressor`. This compressor generates a sparse matrix whose non-zero elements are +1 or -1 (with scaling). More information can be found [here](@ref SparseSign).

We will configure a compression matrix $S$ that compresses the 100 rows of the original system down to 30 rows.

```@example SparseSignExample
# The goal is to compress the 100 rows of A to 30 rows
compression_dim = 30
# We want each row of the compression matrix S to have 5 non-zero elements
non_zeros = 5

# Configure the SparseSign compressor
# - cardinality=Left(): Indicates the compression matrix S will be left-multiplied with A (SAx = Sb).
# - compression_dim: The compressed dimension (number of rows).
# - nnz: The number of non-zero elements per row in S.
# - type: The element type for the compression matrix.
sparse_compressor = SparseSign(
    cardinality=Left(),
    compression_dim=compression_dim,
    nnz=non_zeros,
    type=Float64
)
```
Oops, I suddenly felt 30 rows is not a small enough size, and want to change the dim to 10. Then I can do this:

```@example SparseSignExample
# Change the dimension of the compressor. Similarly, you can use the idea for other configurations' changes.
sparse_compressor.compression_dim = 10
sparse_compressor
```

