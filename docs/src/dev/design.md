# Overview Library Goals
RLinearAlgebra.jl implements randomized numerical linear algebra (RNLA) routines for
two tasks: (1) solving a matrix equations and (2) forming a low-rank approximation 
to matrices.  The primary tool Randomized Linear Algebra uses to accomplish these tasks is
multiplying the large matrix system by a smaller randomized matrix to compress the 
large matrix. In the literature this process if often referred to as sampling or sketching,
RLinearAlgebra.jl refers to this process as compression. 

The library is organized with main techniques falling into one of three types: 
Approximators, Compressors, and Solvers. Solvers feature their own set of 
sub-techniques: Loggers, SolverErrors, and SubSolvers that facilitate solving. Approximators
have only one set of sub-techniques known as ApproximatorErrors.

RLinearAlgebra.jl aims for efficient code with high modularity. These goals are complicated 
by an interdependency between the sub-techniques and the matrix system. RLinearAlgebra.jl
simplifies the dependency by introducing two structures, one that contains user-controlled 
parameters which takes the form of `[Technique]` and a second that is used by the technique 
to complete the techniques task and thus contains the necessary preallocated memory this 
structure takes names of the form `[Technique]Recipe`.

For example, in implementing Gaussian sketching we create a `Gaussian` structure that 
contains the number of rows and columns for the sketch. The user can then construct this 
structure calling the structure name with keyword options. Thus, if the user wanted a 
Gaussian matrix with 3 rows they could create this structure with the call 
`Gaussian(n_rows = 3)`. This structure simply contains the number of rows in the matrix, 
but without knowing information about the matrix that the sketching matrix will be applied 
to, `n_cols` will be unknown. To incorporate information about the matrix the sketch is 
being applied to with the user controlled parameters, we use the function 
`complete_compressor` and generate a `GaussianRecipe` with the appropriate size parameters
and preallocated space to store the Gaussian matrix. 

Once a `[Technique]Recipe` has been created, this data structure can then be used to 
execute a particular technique. The command to execute each technique varies by the class 
of techniques, as such we lay out specifics for each class of techniques in the following 
section.

## Technique Classes
Overall there are three top-level technique classes: (1) Compressors, (2) Solvers, and 
(3) Approximators, with the latter two also having additional sets of technique classes 
used when implementing the top-level techniques. We group the discussions of the technique 
classes by top-level technique.

### Compressors
When implementing any Compressor, RLinearAlgebra requires an immutable `Compressor` 
structure, a mutable `CompressorRecipe` structure, a `complete_compressor` function, a 
`update_compressor!` function, and four mul! functions (two for the compressor 
applied to vectors and two for the sketch applied to matrices). 

#### Compressor Structure
Every compression technique needs a place to store user-defined parameters. 
This will be accomplished by the immutable Compressor structure. 
We present an example structure used for the Sparse Sign technique in ADD CITATION.

```
struct SparseSign <: Compressor
    n_rows::Int64
    n_cols::Int64
    nnz::Int64
end
```
You will first notice that `n_rows` and `n_cols` are fields present in the Compressor, 
these fields allow for the user to specify either the number of rows or the number of 
columns they wish the compressor to have. **Both `n_rows` and `n_cols` are required for 
every Compressor structure.** Beyond those fields the method will
dictate the other parameters that can be made available to the user. In implementation, we 
should aim to make every parameter associated with the compression technique available to 
the user. In addition to the structure itself their should be a **constructor for the 
structure that accepts keyword inputs for each field of the Compressor structure.**
For example in the `SparseSign` case we define,
```
function SparseSign(;n_rows::Int64 = 0, n_cols::Int64 = 0, nnz::Int64 = 8)
    # Partially construct the sparse sign datatype
    return SparseSign(n_rows, n_cols, nnz)
end
```  
#### CompressorRecipe Structure
Because the `Compressor` requires no information about the linear system, 
when the linear system is known, we need a structure that contains preallocated memory for 
the compressor. This location is the `CompressorRecipe`. Again we present an example for
such a structure based on the SparseSign technique:
```
mutable struct SparseSignRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
    max_idx::Int64
    nnz::Int64
    scale::Float64
    idxs::Vector{Int64}
    signs::Vector{Bool}
end
```
Here we have the **required `n_rows` and `n_cols`** fields as fields specific to the 
sketching technique like signs and the number of non-zero elements. 

#### complete_compressor
To create the CompressorRecipe from linear system information and the user implemented 
parameters, we use the function `complete_compressor(::Compressor, ::AbsractMatrix)`, 
if vector information is required we can also define 
`complete_compressor(::Compressor, ::AbsractMatrix, ::AbstractVector)`. 
An example of how this is done for the sparse sign case can be seen below.
```
function complete_compressor(sparse_info::SparseSign, A::AbstractMatrix)
    n_rows = sparse_info.n_rows
    n_cols = sparse_info.n_cols
    # FInd the zero dimension and set it to be the dimension of A
    if n_rows == 0 && n_cols == 0
        # by default we will compress the row dimension to size 2
        n_cols = size(A, 1)
        n_rows = 2
        # correct these sizes
        initial_size = max(n_rows, n_cols)
        sample_size = min(n_rows, n_cols)
    elseif n_rows == 0 && n_cols > 0
        # Assuming that if n_rows is not specified we compress column dimension
         n_rows = size(A, 2)
         # If the user specifies one size as nonzero that is the sample size
         sample_size = n_cols
         initial_size = n_rows
    elseif n_rows > 0 && n_cols == 0
        n_cols = size(A, 1)
        sample_size = n_rows
        initial_size = n_cols
    else
        if n_rows == size(A, 2)
            initial_size = n_rows
            sample_size = n_cols
        elseif n_cols == size(A, 2)
            initial_size = n_cols
            sample_size == n_rows
        else
            @assert false "Either you inputted row or column dimension must match \\
            the column or row dimension of the matrix."
        end
    end

    nnz = (sparse_info.nnz == 8) ? min(8, sample_size) : sparse_info.nnz
    @assert nnz <= sample_size "Number of non-zero indices, $nnz, must be less than \\ 
        compression dimension, $sample_size."
    idxs = Vector{Int64}(undef, nnz * initial_size)
    start = 1
    for i in 1:initial_size
        # every grouping of nnz entries corresponds to each row/column in sample
        stop = start + nnz - 1
        # Sample indices from the intial_size
        @views sample!(
            1:sample_size, 
            idxs[start:stop], 
            replace = false, 
            ordered = true
        )
        start = stop + 1
    end
    
    # Store signs as a boolean to save memory
    signs = bitrand(nnz * initial_size)
    scale = 1 / sqrt(nnz)
    
    return SparseSignRecipe(n_rows, n_cols, sample_size, nnz, scale, idxs, signs)
end
``` 
The `complete_compressor` function assumes that if the user inputs only `n_rows` or `n_cols` 
in the Compressor structure this is the desired compression dimension. If they input neither 
it users a compression dimension of two and if the input both and neither is consistent with 
a dimension of the inputted linear system it returns and error otherwise it assumes the 
inconsistent dimension is the sketching dimension. Once, the sizes of the compressor have 
been determined it next allocates an initial compressor matrix with the minimal necessary 
memory.

#### update_compressor!
To generate a new version of the compressor we can call the function `update_compressor!`, 
this function simply changes the random components of the CompressorRecipe. In the sparse 
sign case this means updating the nonzero indices and the signs as can be seen in the 
following example code.
```
function update_compressor!(
                            S::SparseSignRecipe, 
                            A::AbstractMatrix, 
                            b::AbstractVector, 
                            x::AbstractVector
                           )
    # Sample_size will be the minimum of the two size dimensions of `S`
    sample_size = min(S.n_rows, S.n_cols)
    initial_size = max(S.n_rows, S.n_cols)
    start = 1
    for i in 1:sample_size
        # every grouping of nnz entries corresponds to each row/column in sample
        stop = start + S.nnz - 1
        # Sample indices from the intial_size
        @views sample!(
            1:sample_size, 
            S.idxs[start:stop], 
            replace = false, 
            ordered = true
        )
        start = stop + 1
    end
    # There is no inplace update of bitrand and using sample is slower
    S.signs .= bitrand(S.nnz * initial_size) 
    return
end
```

#### mul!
The last pieces of code that every compression technique requires are the `mul!` functions. 
For these functions we follow the conventions laid out in the LinearAlgebra library where 
there are five inputs (A, B, C, alpha, beta) and it outputs A = beta * A + alpha * B * C.
The `mul!` functions that should be implemented are two for applying the compression matrix
to vectors, one in standard orientation and one for when the adjoint of the matrix is 
applied to the vector. Additionally, two `mul!` functions should be implemented for when 
the compression matrix is applied to a matrix, one for when the compression matrix, C, is 
applied from the left, i.e. CA, and one for when the compression matrix is applied from the 
right, i.e. AC. 

### Solvers
A Solver technique is any technique that aims to find a vector $$x$$ such that either 
$$Ax = b$$ or $$x = \min_u \|A u - b\|_2^2$$. Solvers rely on compression techniques, 
logging techniques, error techniques, and sub-solver techniques. We first discuss 
implementations requirements for the sub-techniques and then discuss how we can use these 
when creating a solver structure.

#### Loggers
Loggers are structures with two goals (1) log a progress value produced by an error metric
and (2) evaluate whether that error is sufficient for stopping. The user controlled inputs
into this for a logging technique will be contained in the Logger structure.

##### Logger
The Logger structure is where the user inputs any information required 
to logging progress and stopping the method. **The Logger is required to have a field for 
`max_it`, `threshold_info`, and `stopping_criterion`.** The `max_it` field is simply a field
for the maximum number of iterations of the method. The `stopping_criterion` is a function 
that returns a stopping decision based on the information in the LoggerRecipe and the 
information supplied by the user in the `threshold_info` field. We present an example of
the Logger structure for a BasicLogger below
```
struct BasicLogger <: Logger
    max_it::Int64
    collection_rate::Int64
    threshold_info::Union{Float64, Tuple}
    stopping_criterion::Function
end
```

Aside from the required parameters the BasicLogger also features a collection rate parameter
to allow the user to specify how often they wish for the LoggerRecipe to log progress.

##### LoggerRecipe
The LoggerRecipe will contain the user-controlled parameters from the Logger as well as the 
memory for storing the log information. All LoggerRecipes **must contain a `max_it` field 
and a `converged` field,** where the `converged` field is a boolean indicating if the method
has converged. An example of a LoggerRecipe is presented below for the BasicLoggerRecipe. 
This Logger has a vector for the history of the progress metric, a field whose inclusion is
strongly suggested. It also has `record_location` field to keep track of where the next 
observed progress estimate should be placed depending on the `collection_rate`.
```
mutable struct BasicLoggerRecipe{F} <: LoggerRecipe where F<:Function
    max_it::Int64
    err::Float64
    threshold_info::Union{Float64, Tuple}
    iteration::Int64
    record_location::Int64
    collection_rate::Int64
    converged::Bool
    stopping_criterion::F
    hist::Vector{Float64}
end
```

##### complete_logger
As with the other `complete_[type]` this function takes a `Logger` data structure and 
performs the appropriate allocations to generate a `LoggerRecipe`. An example of this 
function for BasicLogger is presented below.
```
function complete_logger(logger::BasicLogger, A::AbstractMatrix)
    # We will run for a number of iterations equal to 3 itmes the number of rows if maxit is
    # not set
    max_it = logger.max_it == 0 ? 3 * size(A, 1) : logger.max_it

    max_collection = Int(ceil(max_it / logger.collection_rate))
    # use one more than max it form collection
    hist = zeros(max_collection + 1)
    return BasicLoggerRecipe{typeof(logger.stopping_criterion)}(max_it,
                                                                0.0,
                                                                logger.threshold_info,
                                                                1,
                                                                1,
                                                                logger.collection_rate,
                                                                false,
                                                                logger.stopping_criterion,
                                                                hist
                                                               )
end
```

##### update_logger!
As with the compressors this is a function that performs an in-place update of the 
LoggerRecipe using the inputted progress metric and iteration of the method. An example of 
the `update_logger!` function for the BasicLoggerRecipe is included below.
```
function update_logger!(logger::BasicLoggerRecipe, err::Float64, iteration::Int64)
    logger.iteration = iteration
    logger.err = err
    if rem(iteration, logger.collection_rate) == 0
        logger.hist[logger.record_location] = err
        logger.record_location += 1
    end
    # Always check max_it stopping criterion
    # Compute in this way to avoid bounds error from searching in the max_it + 1 location
    logger.converged = iteration <= logger.max_it ? logger.stopping_criterion(logger) : 
        false
    return

end
```

##### Stopping Functions
As was noted in the description of the required fields for the `Logger` the user should 
have the opportunity to input a stopping function that should take the input of a 
LoggerRecipe to which it updates the value of the `converged` field if stopping should 
occur. An example, this type of function for threshold stopping, stop when progress metric
falls below a particular threshold is presented below.
```
function threshold_stop(log::LoggerRecipe)
    return log.err < log.threshold_info
end
```
#### SolverErrors
For computing the progress of a solver it is important to include implementations of 
particular error techniques. These typically will be techniques like the residual or 
compressed residual, but could be more complicated techniques like an estimate of backwards
stability. 

##### SolverError
This is a structure that holds user-defined parameters for a progress estimation technique.
For basic techniques like the residual where no user-defined parameters are required this
will simply be an empty structure. We have included an example of a `SolverError` structure
for the residual computations.
```
struct FullResidual <: SolverError

end
```

##### SolverErrorRecipe
This is strcuture containing the user-defined parameters from the `SolverError` as well 
memory allocations of a size determined based on the linear system. An example for a 
residual technique has been included below.

```
mutable struct FullResidualRecipe{V<:AbstractVector} <: SolverErrorRecipe
    residual::V
end
```
##### complete_error
To generate the `SolverErrorRecipe` from the information in the linear system and 
`SolverError` we use the function `complete_error`. An example of this function for the 
residual error technique has been included below.

```
function complete_error(error::FullResidual, A::AbstractMatrix, b::AbstractVector)
    return FullResidualRecipe{typeof(b)}(zeros(size(b,1)))
end
```
##### compute_error
To excute the technique we call the function `compute_error` with the inputs of the 
`SolverErrorRecipe`, `Solver`, coefficient matrix `A`, and constant vector `b`. This 
function then performs the necessary computations to return a single value indication of the
progress of the solver. An example of this for the residual technique that returns the norm-
squared of the residual is included below.  
```
function compute_error(
        error::FullResidualRecipe, 
        solver::KaczmarzRecipe, 
        A::AbstractMatrix, 
        b::AbstractVector
    )::Float64
    copyto!(error.residual, b)
    mul!(error.residual, A, solver.solution_vec, -1.0, 1.0)
    return dot(error.residual, error.residual)
end
```

#### SubSolvers
Although, randomized solvers are used to solve a larger linear system. They typically
rely on using compressors to generate a compressed linear system that can be easily solved 
using standard techniques. The specifics of the 'standard' techniques is typically not 
specified. For instance, if the compressed system is a least squares problem one could solve
this system with a QR algorithm or LSQR and potentially get vastly different performance
results. To allow the user to experiment with different techniques for solving the 
compressed linear system, we introduce the SubSolver data structures.

##### SubSolver
This is a data structure that allows the user to specify how they wish to solve the
compressed linear systems generated in the solving process. When the solver type is a direct
method it is possible for there to be no user inputs in this data structure. For iterative
methods there could be extensive user-defined parameters included in this structure. For 
example, for a LSQR SubSolver the user could input the maximum of iterations, a 
preconditioner type, or stopping thresholds. We have included an example of the `SubSolver`
structure for the `LQSolver`, which is an approach for solving undetermined linear systems 
and does not have any user-defined parameters associated with it.
```
struct LQSolver <: SubSolver

end
```

##### SubSolverRecipe
This is a data structure that contains the preallocated memory necessary for solving the 
linear system. 

##### complete_solver
This is a function that takes a `SubSolver` and the linear system as input and uses these 
inputs to output a SubSolverRecipe.

##### update_sub_solver
This is a function that updates the preallocated memory in the SubSolverRecipe with 
the relevant information for the new compressed linear system. 

##### ldiv!
A function that uses the SubSolverRecipe to solve the compressed linear system.

#### Solvers
With an understanding of all of these sub techniques, we can discuss how to use these 
methods to implement a Solver technique. The first data structure required for a solver is 
the `Solver` structure.

##### Solver
The Solver data structure is a structure where the user can input values of user-defined 
parameters specific to a particular type of solver. This typically involves the user
inputting the structures associated with their desired Compressor, Logger, Error, and 
SubSolver, as well as any parameters like step-sizes associated with the particular 
randomized solver they are using. As an example, we have included the Solver structure 
associated with the Kaczmarz solver. It is important to note that constructors for these
techniques should have keyword inputs with predefined defaults.
```
mutable struct Kaczmarz <: Solver 
    alpha::Float64
    S::Compressor
    log::Logger
    error::SolverError
    sub_solver::SubSolver
end
```

##### SolverRecipe
The SolverRecipe will contain all the preallocated memory associate with the solver, the 
solver specific user-defined parameters, and all recipes associated with the sub-techniques
included in the Solver. We have included an example for the `KaczmarzRecipe` below.
```
mutable struct KaczmarzRecipe{T<:Number, 
                              V<:AbstractVector,
                              M<:AbstractMatrix, 
                              VV<:SubArray,
                              MV<:SubArray,
                              C<:CompressorRecipe, 
                              L<:LoggerRecipe,
                              E<:SolverErrorRecipe, 
                              B<:SubSolverRecipe
                             } <: SolverRecipe
    S::C
    log::L
    error::E
    sub_solver::B
    alpha::Float64
    compressed_mat::M
    compressed_vec::V
    solution_vec::V
    update_vec::V
    mat_view::MV
    vec_view::VV
end
```
The first four fields are associated with the sub-techniques for the solver. The alpha 
field is a user defined value and the remaining fields are preallocated space for storing
the result of the compression and the solution vector.

##### complete_solver
The `complete_solver` function performs the necessary computations and allocations to change
a `Solver` structure into a `SolverRecipe`. In the example code below for a Kaczmarz solver
these computations include running `complete_[technique]` for the compression, logging, 
error, and sub solver techniques, as well as allocating memory for storing the compressed 
matrix and compressed vector, the solution vector, and update vector. **The views allocated
by this function should be replicated in other solver structures to allow for varying sizes
of the compression matrix.**
```
function complete_solver(
        solver::Kaczmarz, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )
    # Dimension checking will be performed in the complete_compressor
    compressor = complete_compressor(solver.S, A, b)
    logger = complete_logger(solver.log, A, b)
    error = complete_error(solver.error, A, b) 
    # Check that required fields are in the types
    @assert isdefined(error, :residual) "ErrorRecipe $(typeof(error)) does not contain the\ 
field 'residual' and is not valid for a kaczmarz solver."
    @assert isdefined(logger, :converged) "LoggerRecipe $(typeof(logger)) does not contain\
 the field 'converged' and is not valid for a kaczmarz solver."
    # Assuming that max_it is defined in the logger
    alpha::Float64 = solver.alpha 
    # We assume the user is using compressors to only decrease dimension
    n_rows::Int64 = compressor.n_rows
    n_cols::Int64 = compressor.n_cols
    sample_size = n_rows
    initial_size = n_cols
    rows_a, cols_a = size(A)
    # Allocate the information in the buffer using the types of A and b
    compressed_mat = typeof(A)(undef, sample_size, cols_a)
    compressed_vec = typeof(b)(undef, sample_size) 
    # Since sub_solver is applied to compressed matrices use here
    sub_solver = complete_sub_solver(solver.sub_solver, compressed_mat, compressed_vec)
    mat_view = view(compressed_mat, 1:sample_size, :)
    vec_view = view(compressed_vec, 1:sample_size)
    solution_vec = x
    update_vec = typeof(x)(undef, cols_a)
    return KaczmarzRecipe{eltype(A), 
                          typeof(b), 
                          typeof(A), 
                          typeof(vec_view),
                          typeof(mat_view),
                          typeof(compressor),
                          typeof(logger),
                          typeof(error),
                          typeof(sub_solver)
                         }(compressor, 
                           logger, 
                           error,
                           sub_solver,
                           alpha,
                           compressed_mat,
                           compressed_vec,
                           solution_vec,
                           update_vec,
                           mat_view,
                           vec_view
                          )
end
```

##### rsolve!
Every implementation of a Solver technique should include a `rsolve!` function that performs
in-place updates to a solution vector and `SolverRecipe`. An example of such an 
implementation for a Kaczmarz solver is included below. To the greatest extent possible
the implementation should be written in a way that avoids new memory allocations. This means
making use in-place update functions like `mul!` or `ldiv!` rather than `*` or `\`.
```
function rsolve!(
        solver::KaczmarzRecipe, 
        x::AbstractVector, 
        A::AbstractMatrix, 
        b::AbstractVector
    )
    solver.solution_vec = x
    err = 0.0
    for i in 1:solver.log.max_it
        err = compute_error(solver.error, solver, A, b)
        # Update log adds value of err to log and checks stopping
        update_logger!(solver.log, err, i)
        if solver.log.converged
            return solver.solution_vec, solver.log
        end

        # generate a new version of the compression matrix
        update_compressor!(solver.S, A, b, x)
        # based on size of new compressor update views of matrix
        # this should not result in new allocations
        rows_s, cols_s =  size(solver.S)
        solver.mat_view = view(solver.compressed_mat, 1:rows_s, :)
        solver.vec_view = view(solver.compressed_vec, 1:rows_s)
        # compress the matrix and constant vector
        mul!(solver.mat_view, solver.S, A)
        mul!(solver.vec_view, solver.S, b)
        # Compute the block residual
        mul!(solver.vec_view, solver.mat_view, solver.solution_vec, -1.0, 1.0)
        # sub-solver needs to designed for new compressed matrix
        update_sub_solver!(solver.sub_solver, solver.mat_view)
        # use sub-solver to find update the solution
        sub_solve!(solver.update_vec, solver.sub_solver, solver.vec_view)
        # Using over-relaxation parameter, alpha, to update solution
        solver.solution_vec .+= solver.alpha .* solver.update_vec 
    end

    return solver.solution_vec, solver.log
end
```

### Approximators
Aside from solving linear systems, Randomized Linear Algebra has also been proven to be
extremely useful for generating low rank approximations to linear systems. These low-rank 
approximations can then be used to solve linear systems or perform more efficient
matrix-matrix multiplications. The main types of low rank approximation methods implemented
in this version of the library are random range finder techniques like random SVD, CUR 
type methods, and Nystrom Methods. Low rank approximations can be formed simply by calling 
the `rapproximate` function. Once a Low-rank approximation has been formed it can then be 
applied either as a preconditioner by calling the `ldiv!` function or multiplied by calling 
the `mul!` function. Each of the low rank approximation technique requires the
implementation of the following data structures and functions.

#### Approximator
This is a data structure that contains the user defined parameters for an approximator. An 
example of this structure for the RangeFinder decomposition is included below.
```
mutable struct RangeFinder <: Approximator
    S::Compressor
    error::ErrorMethod
end
```

#### ApproximatorRecipe
This a data structure that contains preallocated memory and the user-defined parameters for 
a specific approximation method. An example of this data structure for a RangeFinder 
decomposition is included below.
```
mutable struct RangeFinderRecipe <: ApproximatorRecipe
    S::CompressorRecipe
    error::ErrorMethodRecipe
    compressed_mat::AbstractMatrix
    approx_range::AbstractMatrix
end
```

#### complete_approximator
A function that takes the linear system information through the matrix `A` and the 
`Approximator` data structure to output an `ApproximatorRecipe` with properly allocated 
storage for the low-rank approximation. An example of this function for the 
`RangeFinderRecipe` is included below.
```
function complete_approximator(approx::RangeFinder, A::AbstractMatrix)
    S = complete_compressor(approx.S, A)
    err = complete_error(approx.error, A)
    s_rows, s_cols = size(S)
    a_rows, a_cols = size(A)
    compressed_mat = Matrix{eltype(A)}(undef, a_rows, s_cols)
    approx_range = Matrix{eltype(A)}(undef, a_rows, s_cols) 
    return RangeFinderRecipe(S, err, compressed_mat, approx_range)
end
```

#### r_approximate
A function that returns an `ApproximatorRecipe` and approximation error value for a
particular approximation method. The returned `ApproximatorRecipe` can then be used for 
matrix multiplication or preconditioning through the use of the `mul!` and `ldiv!` functions
respectively. An example of this function for the `RangeFinderRecipe` is included below.
```
function r_approximate(
    approximator::RangeFinderRecipe
    A::AbstractMatrix
)
    m, n = size(A)
    update_compressor!(aproximator.S)
    # compuress the matrix
    mul!(compressed_mat, A, aproximator.S)
    # Array is required to compute the skinny qr
    approximator.approx_range .= Array(qr(compressed_mat).Q)
    err = compute_error(aproximator.error, A)
    return approximator, error
end
```

#### ldiv!
A function that solves the system `Mx = b` for `x` where M is a low rank approximation 
matrix. This is useful for preconditioning linear systems. When there is no obvious way to
use the low rank approximation to solve this system the implementation will be the same as
the implementation for `mul!`.

#### mul!
A function that multiplies a low rank approximation with a matrix.

#### ApproximatorError
A data structure that takes the user controlled parameters for a method that computes the 
approximation error, e.g `A - QQ'A` for a particular approximation method.  
```
mutable struct ProjectedError <: ApproximatorError

end

```

#### ApproximatorErrorRecipe
A data structure that takes the user controlled parameters and preallocated memory for a 
method that computes the approximation error, e.g `A - QQ'A` for a particular approximation 
method.  
```
mutable struct ProjectedErrorRecipe{T, M{T}} <: ApproximatorErrorRecipe 
    where M <: AbstractMatrix
    error::Float64
    large_buff_mat::M
    small_buffer_mat::M
end
```
#### complete_error
A function that takes the information from a `ApproximatorError`, `CompressorRecipe`, and an
`AbstractMatrix` to create an `ApproximatorErrorRecipe`. An example for the 
`ProjectedError` structure is included below.
```
function complete_error(error::ProjectedError, S::CompressorRecipe, A::AbstractMatrix)
    row_s, col_s = size(S)
    row_a, col_a = size(A)
    T = eltype(A)
    M = Matrix{T}
    small_buffer_mat = M(undef, col_s, col_a) 
    large_buffer_mat = M(undef, row, col_a) 
    return ProjectedErrorRecipe{T, M}(0.0, large_buffer_mat, small_buffer_mat)
end
```

#### compute_error
A function that computes the error of a particular approximation method with respect to the 
matrix `A` for a particular approximation technique. An example for `ProjectedError` is 
included below.
```
function compute_error(
        error::ProjectedErrorRecipe,
        approximator::RandRangeFinderRecipe,
        A::AbstractMatrix
    )
    mul!(error.small_buffer_mat, approximator.row_space', A)
    mul!(error.large_buffer_mat, approximator.row_space, error.small_buffer_mat)
    error.large_buffer_mat .-= A
    error.error = norm(error.large_buffer_mat)
    return error.error
end
```
