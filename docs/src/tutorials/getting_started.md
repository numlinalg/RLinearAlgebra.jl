# A simple example


## Use Case: Solving a Least-Squares Problem with the Sparse Sign Method

This guide demonstrates how to use the `SparseSign` compression method from the `RLinearAlgebra.jl` package to solve an overdetermined linear system (i.e., a least-squares problem) of the form:
$$
\min_{x} \|Ax - b\|_2^2
$$
We will follow the design philosophy of `RLinearAlgebra.jl` by composing different modules (`Solver`, `Compressor`, `Logger`, etc.) to build and run the solver.

---
### 1. Problem Setup

First, we define a specific linear system $Ax = b$. To verify the accuracy of the final result, we will first create a known solution, $x_{\text{true}}$, and then use it to generate the vector $b$.

* **Matrix `A`**: A random $100 \times 20$ matrix.
* **Vector `b`**: Calculated as $b = A x_{\text{true}}$, with dimensions $100 \times 1$.
* **Goal**: Find a solution $x$ that is as close as possible to $x_{\text{true}}$.

---
### 2. Solution Steps

We will proceed through the following steps to build and run a randomized solver.

#### Step 1: Environment Setup and Problem Definition
First, we need to import the required libraries and create the matrix `A` and vector `b` as defined above. We will also set an initial guess, `x_init`, for the solver.

```julia
# Import relevant libraries
using RLinearAlgebra, Random, LinearAlgebra


# Define the dimensions of the linear system
num_rows, num_cols = 100, 20

# Create the matrix A and a known true solution x_true
A = randn(Float64, num_rows, num_cols)
x_true = randn(Float64, num_cols)

# Calculate the right-hand side vector b from A and x_true
b = A * x_true

# Set an initial guess for the solution vector x (typically a zero vector)
x_init = zeros(Float64, num_cols)

println("Problem setup complete:")
println(" - Dimensions of matrix A: ", size(A))
println(" - Dimensions of vector b: ", size(b))
```

#### Step 2: Configure the `SparseSign` Compressor
The core idea of randomized methods is to reduce the scale of the original problem using a random "sketch" or "compression" matrix, $S$. Here, we choose `SparseSign` as our `Compressor`. This compressor generates a sparse matrix whose non-zero elements are +1 or -1 (with scaling).

We will configure a compression matrix $S$ that compresses the 100 rows of the original system down to 30 rows.

```julia
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



