# Use Case: Solving a Least-Squares Problem with the Sparse Sign Method

This guide demonstrates how to use the `SparseSign` compression method from the `RLinearAlgebra.jl` package to solve an overdetermined linear system (i.e., a least-squares problem) of the form:

$$\min_{x} \|Ax - b\|_2^2$$

We will follow the design philosophy of `RLinearAlgebra.jl` by composing different modules (`Solver`, `Compressor`, `Logger`, etc.) to build and solve the problem.

---
## 1. Problem Setup

Let's define a specific linear system $Ax = b$. 

To verify the accuracy of the final result, suppose that we know the true solution of the system, $x_{\text{true}}$, and then use it and a random generated matrix $A$ to generate the vector $b$.

* **Matrix `A`**: A random $1000 \times 20$ matrix.
* **Vector `b`**: Calculated as $b = A x_{\text{true}}$, with dimensions $1000 \times 1$.
* **Goal**: Find a solution $x$ that is as close as possible to $x_{\text{true}}$.

To achieve this, we need to import the required libraries and create the matrix `A` and vector `b` as defined above. We will also set an initial guess, `x_init`, for the solver.

```@example SparseSignExample
# Import relevant libraries
using RLinearAlgebra, Random, LinearAlgebra


# Define the dimensions of the linear system
num_rows, num_cols = 1000, 20

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
## 2. Create compressors

In practice, we may encounter a much larger $A$ matrix than what we have here. Solving the problem with such a large matrix can slow down the performance of iterative algorithms that we will use to solve the least square problem. Therefore, we will use a randomized sketching technique to compress the matrix A and the corresponding vector b to a lower dimension, while preserving the essential geometric information of the system.

Here, we will use the sparse sign method.

### (a) Configure the `SparseSign` Compressor

The idea of randomized methods is to reduce the scale of the original problem when the dimention of matrix $A$ is too big, using a random "sketch" or "compression" matrix, $S$. Here, we choose `SparseSign` as our `Compressor`. This compressor generates a sparse matrix whose non-zero elements are +1 or -1 (with scaling). More information can be found [here](@ref SparseSign).

We will configure a compression matrix `S` that compresses the 100 rows of the original system down to 30 rows.

```@example SparseSignExample
# The goal is to compress the 1000 rows of A to 300 rows
compression_dim = 300
# We want each row of the compression matrix S to have 5 non-zero elements
non_zeros = 5

# Configure the SparseSign compressor
# - cardinality=Left(): Indicates the compression matrix S will be 
#    left-multiplied with A (SAx = Sb).
# - compression_dim: The compressed dimension (number of rows).
# - nnz: The number of non-zero elements per column (left)/row (right) in S. 
# - type: The element type for the compression matrix.
sparse_compressor = SparseSign(
    cardinality=Left(),
    compression_dim=compression_dim,
    nnz=non_zeros,
    type=Float64
)
```

---
### (b) Build the `SparseSign` recipe

After configuring the compressor, we need to combine it with our specific matrix `A` to create a `SparseSignRecipe`. This recipe contains the generated sparse matrix and all necessary information to perform the compression efficiently.

```@example SparseSignExample
# Pass the compressor configuration and the original matrix A to
# create the final compression recipe.
S = complete_compressor(sparse_compressor, A)

# You can closely look at the compression recipe you created.
println("Configurations of compression matrix:")
println(" - Compression matrix is applied to left or right: ", S.cardinality)
println(" - Compression matrix's number of rows: ", S.n_rows)
println(" - Compression matrix's number of columns: ",  S.n_cols)
println(" - The number of nonzeros in each column (left)/row (right) of compression matrix: ",  S.nnz)
println(" - Compression matrix's nonzero entry values: ",  S.scale)
println(" - Compression matrix: ",  S.op)
```
If the compression dimension of `300` rows is considered too large, it can be changed to `10` by updating the compressor configuration and rebuilding the recipe as follows:
```@example SparseSignExample
# Change the dimension of the compressor. Similarly, you can use the same idea 
# for other configurations' changes.
sparse_compressor.compression_dim = 10

# Rebuild the compressor recipe
S = complete_compressor(sparse_compressor, A)
println("Compression matrix's number of rows: ", S.n_rows)
```

### (c) Apply the sparse sign matrix to the system

While the solver can use the `S` recipe to perform multiplications on-the-fly, it can sometimes be useful to form the compressed system explicitly. We can use `*` for this.

```@example SparseSignExample
# Form the compressed system SAx = Sb
SA = S * A
Sb = S * b

println("Dimensions of the compressed system:")
println(" - Matrix SA: ", size(SA))
println(" - Vector Sb: ", size(Sb))
```

---
## 3. Create solver

With the problem and compressor defined, the next step is to choose and configure a solver. Here, we choose to use the 
[Kaczmarz solver](@ref Kaczmarz). We configure it by passing in "ingredient" objects for each of its main functions: compressing the system (already done), logging progress, and checking for errors.

### (a)  Configure the logger and stopping rules

To monitor the solver, we will use a `BasicLogger`. This object will serve two purposes: record the error history, and tell the solver when to stop.

We will configure it to stop after a maximum of `50` iterations or if the calculated error drops below a tolerance of `1e-6`. And we use `collection_rate = 5` 
to configure the frequence of error recording to be every $5$ steps.

```@example SparseSignExample
# Configure the logger to control the solver's execution
logger = BasicLogger(
    max_it = 50,
    threshold = 1e-6,
    collection_rate = 5
)
```

---
### (b) Build the Kaczmarz Solver
Now, we assemble our configured components (compressor `S`, logger `L`) into the main Kaczmarz solver object. We will use the default methods for error checking and the sub-solver to be LQ factorization ([LQSolver](@ref LQSolver)).

```@example SparseSignExample
# Create the Kaczmarz solver object by passing in the ingredients
kaczmarz_solver = Kaczmarz(
    compressor = sparse_compressor,
    log = logger,
    sub_solver = LQSolver()
)
```
Before we can run the solver, we must call `complete_solver`. This function takes the solver configurations and the specific problem data `A, b, x_init` and creates a `KaczmarzRecipe`. The recipe pre-allocates all the necessary memory buffers for efficient computation.

```@example SparseSignExample
# Create the solver recipe by combining the solver and the problem data
solver_recipe = complete_solver(kaczmarz_solver, x_init, A, b)
```

---
## 4. Solve the Compressed System

With the recipe fully prepared, we can now call `rsolve!` to run the Kaczmarz algorithm. The function will iterate until the `stopping criterion` in the `logger` is met.

The `rsolve!` function will modify `x_init` in-place, updating it with the calculated solution.

```@example SparseSignExample
# Run the solver!
rsolve!(solver_recipe, x_init, A, b)

# The solution is now stored in the updated x_init vector
solution = x_init;
```

---
## 5. Verify the result

Finally, let's check how close our calculated solution is to the known `x_true`. We can do this by calculating the Euclidean norm of the difference between the two vectors. A small error norm indicates a successful approximation.

```@example SparseSignExample
# We can inspect the logger's history to see the convergence
error_history = solver_recipe.log.hist;
println(" - Solver stopped at iteration: ", solver_recipe.log.iteration)
println(" - Final error: ", error_history[solver_recipe.log.record_location - 1])

# Calculate the norm of the error
error_norm = norm(solution - x_true)
println(" - Norm of the error between the solution and x_true: ", error_norm)
```
As you can see, by using the modular Kaczmarz solver, we were able to configure a randomized block-based method and find a solution vector that is very close to the true solution. 