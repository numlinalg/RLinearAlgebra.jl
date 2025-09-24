# A library for exploring Randomized Linear Algebra
If you are here, you probably know that Linear Algebra is foundational to data science and 
scientific computing. You also probably know Linear Algebra routines dominate the 
computational cost of many of the algorithms in these fields. Thus, improving the 
scalability of these algorithms requires more scalable Linear Algebra techniques.

An exciting new set of techniques that offers such improved scalability of 
Linear Algebra techniques are Randomized Linear Algebra techniques. 
In general, Randomized Linear Algebra techniques aim to achieve this improved
scalability by forming a representative sample of a matrix and performing 
operations on that sample. In some circumstances operating on this sample
can offer profound speed-ups as can see in the following example 
where a technique known as the RandomizedSVD (see [halko2011finding](@cite)) 
is used to compute a rank-20 approximation to ``3000 \times 3000`` matrix 
``A`` in place of a truncated SVD. Compared to computing the SVD and 
truncating it, the RandomizedSVD is 100 times faster and just as accurate 
as the truncated SVD.

```julia
using RLinearAlgebra
using LinearAlgebra

# Generate a rank-20 matrix
A = randn(3000, 20) * randn(20, 3000);

@time U,S,V = svd(A);
#    4.566639 seconds (13 allocations: 412.354 MiB, 0.92% gc time)

# Form the RandomizedSVD data structure
technique = RandomizedSVD(
    compressor = Gaussian(compression_dim= 22,  cardinality=Right()), 
    orthogonalize=false, 
    power_its = 0
)

# Take the RandomizedSVD of A
@time rec = rapproximate(technique, A);
#    0.050950 seconds (39 allocations: 5.069 MiB)

# Take the norm of the difference between the RandomizedSVD and TrunctatedSVD at rank 22
norm(rec.U * Diagonal(rec.S) * rec.V' - U[:,1:22] * Diagonal(S[1:22]) * (V[:, 1:22])')
# 6.914995919005829e-11
```

Over the years, numerous Randomized Linear Algebra approaches have been proposed not only
for basic linear tasks such as computing a low-rank approximation to a matrix, solving 
a linear system, or solving a least squares problem, but also for how obtain a 
representative sample of the matrix itself. To this point, a single easy to prototype 
library has not been developed to bring these techniques to the masses. RLinearAlgebra.jl is designed to be exactly such an easy-to-use library.

In particular, RLinearAlgebra.jl leverages a modular design to allow you 
to easily test Randomized Linear Algebra routines under a wide-range of parameter choices.  RLinearAlgebra.jl provides routines for two core Linear Algebra tasks: finding a solution to
a linear system via ``Ax=b`` or ``\min_x \|Ax - b\|`` and forming a low rank 
approximation to a matrix, ``\hat A`` where ``\hat A \approx A``. The solution to a linear
system appears everywhere: Optimization, Tomography, Statistics, Scientific Computing, Machine
Learning, etc. The low-rank approximation problem has only become more relevant in recent years
owing to the drastic increase in matrix sizes. It has been widely used in Statistics via PCA, but 
also has become increasingly more relevant in all the fields where solving a linear system is 
relevant. 

This manual will walk you through the use of the RLinearAlgebra.jl library. The remainder of this
section will be focused on providing an overview of the common design elements in the library, 
and information about how to get started using the library.

## Overview of the Library
The library is based on two data structure types: **techniques** that contain the parameters 
that define a particular method and **technique recipes** that contain these parameters and 
the necessary preallocations for the desired technique to be executed efficiently. As the 
user you only need to define the techniques and the library will do all the work to form
the recipes for you. If you wish to convert a technique into a technique recipe you can use
the `complete_[technique type]` function.

### The Technique Types
With an understanding of the basic structures in the library, one may wonder, what 
types of techniques are there? First, there are the techniques for solving the linear 
system, `Solvers` and techniques for forming a low-rank approximation to a matrix, 
`Approximators`. Both `Solvers` and `Approximators` achieve speedups by working on 
compressed forms (often known as sketched or sampled) of the linear system, techniques that 
compress the linear system are known as `Compressors`. Aside from these global techniques, 
there are also techniques that are specific to `Solvers`, which include: 

1. `SubSolvers`, techniques that solve the inner (compressed) linear system.
2. `Loggers`, techniques that log information and determine whether a stopping criterion has
    been met.
3. `SolverError`, a technique that computes the error of a current iterate of a solver.  

Similarly, `Approximators` have their own specific techniques, which include:

1. `ApproximorError`, a technique that computes the error of an `Approximator`.

With all these technique structures, you may be wondering, what functions
can I call on these structures? Well, the answer is not many. As is 
summarized in the following table.  

| Technique         | Parent Technique   | Function Calls                        |  
| ----------        | ------------------ | -----------------------------------   |  
|`Approximator`     | None               | `complete_approximator`,`rapproximate`|
|`Compressor`       | None               | `complete_compressor`                 |  
|`Solver`           | None               | `complete_solver`, `rsolve`           |
|`ApproximatorError`| `Approximator`     | `complete_approximator_error`         |
|`Logger`           | `Solver`           | `complete_logger`                     |
|`SolverError`      | `Solver`           | `complete_solver_error`               |
|`SubSolver`        | `Solver`           | `complete_sub_solver`                 |


From the above table we can see that essentially all you are able to do unless you are using 
an `Approximator` or a `Solver` is complete the technique. The reason being that all the 
technique structures contain only information about algorithmic parameters that require no 
information about the linear system. The recipes on the other hand have all the information 
required to use a technique including the required pre-allocated memory. We determine the 
preallocations for the Recipes by merging the parameter information of the technique 
structures with the matrix and linear system information via the `complete_[technique]` 
functions, which is the only function that you can call when you have a technique structure. 
There is a special exception for `rsolve` and `rapproximate` because they implicitly call 
all the necessary completes to form the appropriate recipe. The bottom line is that do 
anything useful you will need a recipe.

### The Recipe Types
Every technique can be transformed into a recipe. As has been stated before, what makes the 
recipes different is that they contain all the required memory allocations. These allocations can
only be determined from once the matrix is known. As a user, 
all you have to know is that as soon as you have a recipe you can do a lot. As can be seen 
in the following table.

| Technique Recipe  | Parent Recipe | Function Calls                      |  
|-----------------  |------------------| ---------------------------------|
|`Approximator`     | None             | `mul!`, `rapproximate!`          |
|`Compressor`       | None             | `mul!`,`update_compressor!`      |
|`Solver`           | None             | `rsolve!`                        |
|`ApproximatorError`| `Approximator`   | `compute_approximator_error`     |
|`Logger`           | `Solver`         | `reset_logger!`, `update_logger!`|
|`SolverError`      | `Solver`         | `compute_error`                  |
|`SubSolver`        | `Solver`         | `update_sub_solver!`,`ldiv!`     |

Instead of providing 
a different function for each method associated with these tasks, RLinearAlgebra.jl 
leverages the multiple-dispatch functionality of Julia to allow all linear systems and 
least squares problems to be solved calling the function 
`rsolve(solver::Solver, x::AbstractVector, A::AbstractMatrix, b::AbstractVector)` 
and all matrices to be approximated by calling the function 
`rapproximate(approximator::Approximator, A::AbstractMatrix)`. Under this design, changing 
the routine for solving your linear system or approximate your matrix is as 
simple as changing the`solver` or `approximator` arguments. 

## Installing RLinearAlgebra
Currently, RLinearAlgebra.jl is not registered in Julia's official package registry. 
It can be installed by writing in the REPL:
```julia
] add https://github.com/numlinalg/RLinearAlgebra.jl.git
```
It can also be cloned into a local directory and installed by:
1. `cd` into the local project directory 
2. Call `git clone https://github.com/numlinalg/RLinearAlgebra.jl.git`
3. Run Julia
4. Call `using Pkg`
5. Call `Pkg.activate(RLinearAlgebra.jl)`
6. Call `Pkg.instantiate()`

For more information see [Using someone else's project](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project).
 

 
## Using RLinearAlgebra.jl
For this example let's assume that we have a vector that we wish to compress
using one the RLinearAlgebra.jl `SparseSign` compressor. To do this: 

```@raw html
<ol>
  <li>
    Load <code>RLinearAlgebra.jl</code> and generate your vector
<pre><code class="language-julia">using RLinearAlgebra
using LinearAlgebra
# Specify the size of the vector
n = 10000
x = rand(n)</code></pre>
  </li>
  <li>
    Define the <code>SparseSign</code> technique. This requires us to specify a <code>cardinality</code>,
    the direction we intend to apply the compressor from, and a <code>compression_dim</code>, 
    the number of entries we want in the compressed vector. In this instance we 
    want the cardinality to be <code>Left()</code> and the <code>compression_dim = 20</code>.
    <pre><code class="language-julia">comp = SparseSign(compression_dim = 20, Cardinality = Left())</code></pre>
  </li>
  <li>
    Use the <code>complete_compressor</code> function to generate the <code>SparseSignRecipe</code>.
    <pre><code class="language-julia">S = complete_compressor(comp, A)</code></pre>
  </li>
  <li>
    Apply the compressor to the vector using the multiplication function
    <pre><code class="language-julia">Sx = S * x
    
norm(Sx)

norm(x)</code></pre>
  </li>
</ol>
```




```@raw html
<!--- !!!please add contents before this line!!!
Do not know why the comment will just comment all the codes followed after them.

## Using RLinearAlgebra.jl
For this example let's assume that we have a vector that we wish to compress
using one the RLinearAlgebra.jl `SparseSign` compressor. To do this: 

1. Load RLinearAlgebra.jl and generate your vector
```julia
using RLinearAlgebra
using LinearAlgebra
# Specify the size of the vector
n = 10000
x = rand(n)  
```
2. Define the `SparseSign` technique. This requires us to specify a `cardinality`,
    the direction we intend to apply the compressor from, and a `compression_dim`, 
    the number of entries we want in the compressed vector. In this instance we 
    want the cardinality to be `Left()` and the `compression_dim = 20`.
```julia
comp = SparseSign(compression_dim = 20, Cardinality = Left())
```
1. Use the `complete_compressor` function to generate the `SparseSignRecipe`.
```julia
S = complete_compressor(comp, A)
```
1. Apply the compressor to the vector using the multiplication function
```julia
Sx = S * x

norm(Sx)

norm(x)
```

### Keep Markdown
For this example let's assume that we have a vector that we wish to compress
using one the RLinearAlgebra.jl `SparseSign` compressor. To do this: 

1. Load RLinearAlgebra.jl and generate your vector

        using RLinearAlgebra
        using LinearAlgebra
        # Specify the size of the vector
        n = 10000
        x = rand(n)  

2. Define the `SparseSign` technique. This requires us to specify a `cardinality`,
    the direction we intend to apply the compressor from, and a `compression_dim`, 
    the number of entries we want in the compressed vector. In this instance we 
    want the cardinality to be `Left()` and the `compression_dim = 20`.

        comp = SparseSign(compression_dim = 20, Cardinality = Left())

3. Use the `complete_compressor` function to generate the `SparseSignRecipe`.

        S = complete_compressor(comp, A)

1. Apply the compressor to the vector using the multiplication function

        Sx = S * x

        norm(Sx)

        norm(x)
-->
```
