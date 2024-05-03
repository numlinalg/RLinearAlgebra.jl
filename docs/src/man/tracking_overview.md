# Tracking an Iterative Sketching Solver
***Here we describe a tracking procedure for the row sketching method. The same descriptions and instructions also apply for the column sketching methods, but use the residual of the normal equations rather than the residual of the linear system***

## The need for tracking
When using an iterative row sketching method like Randomized Block Kaczmarz to find a $x^*$ such that $Ax^*=b$ for $A\in \mathbb{R}^{m\times n}$, $b\in \mathbb{R}^n$,
knowing the quality of the solution is integral for being able to make appropriate stopping decisions. This solution quality can be determined by progress tracking procedures, the most common one being computing the norm squared of the residual of the linear system, $\|Ax_k-b\|_2^2$. Without careful thought progress tracking can undermine the benefits of the iterative sketching procedure. For instance, the update to the sketch is computed using only a sketch of the matrix, $SA \in \mathcal{R}^{p \times n}$, $p \ll m$, and constant vector $Sb \in \mathcal{R}^p$; however, the residual computation requires the accessing the full matrix $A$ to compute. This completely undermines the memory benefits of working with the smaller sketch. Additionally, if $p \ll \sqrt{m}$, which is often the case in practice we can see in the table below that the computational complexity of the update is less than computing the residual.

| Action | Cost | Accesses to Full Matrix|
|--------|------| :-----------: |
|Update| $\mathcal{O}(np^2)$| 0|
|Residual | $\mathcal{O}(nm)$| 1|
|Sketched MA |   $\mathcal{O}(np)$ | 0|

These cost could be avoided by computing the sketched
residual, i.e. $\|S A x - Sb\|_2^2$, but at the cost exact knowledge of 
progress. To reduce this randomness Pritchard and Patel propose using 
a moving average of the sketched residuals in "Solving, Tracking 
and Stopping Streaming Linear Inverse Problems." Specifically they define
the progress estimator 
$$
    \hat \rho_k^\lambda = \sum_{i-\lambda +1}^k \frac{\|S_i A x - S_ib\|_2^2}{\lambda},
$$  
where $\lambda$ is the width of the moving average window. 


This tracking technique can be used in RLinearAlgebra.jl with the log option
[`LSLogFullMA()`](@ref) when defining the solver, with a default moving average width of 30.
```julia
using RLinearAlgebra

# Generate a system
A = rand(20, 5);
x = rand(5);
b = A * x;

# Specify solver
solver = RLSSolver(
    LinSysVecRowRandCyclic(),   # Random Cyclic Sampling
    LinSysVecRowProjStd(),      # Hyperplane projection
    LSLogFullMA(),              # Full Logger: maintains moving average residual history
    LSStopMaxIterations(200),   # Maximum iterations stopping criterion
    nothing                     # System solution (not solved yet)
);

# Solve the system
sol = rsolve(solver, A, b)
```
## Understanding how to use the estimator
The user is able to choose their own width of the moving average by inputting 
`lambda2=USER_WIDTH` as an option within [`LSLogFullMA()`](@ref). Increasing the width
will decrease the variability of the progress estimate and therefore is a suggested
action when the matrix is poorly conditioned or has highly variable magnitude of row 
norms. However, in most cases the default option of 30 should be  sufficient for reasonable tracking. 

In most cases, it is desirable to in fact use two widths for 
the moving average estimator because often at early iterations
much of the observed variability in the residuals arises from the 
fast convergence of the algorithm. As this variability is primarily because of convergence properties rather than randomness, it is undesirable to smooth this out. Thus, it makes 
sense in this phase to use a smaller moving average width, $\lambda_1$, which should typically be set to 1 (the user can change using the option `lambda1=SMALLER_WIDTH` in [`LSLogFullMA()`](@ref)). Once the 
algorithm leaves this fast convergence phase it, then the wider moving average window should be used. This switch is determined to
be the point where there is no longer monotonic decreases in the norm of the sketched residual. 

If the user wants to compare the performance of the moving average of the sketched residuals to 
that of the true residuals then it is possible to input the option `true_res=true` into [`LSLogFullMA()`](@ref). This will 
perform the same moving average tracking procedure, but use the true residual rather than the sketched one.

Finally, if the user wants to get uncertainty sets for the sketched residual tracking they can use
the function [`get_uncertainty()`](@ref), whose use is demonstrated below. 
For the uncertainty sets, the user can specify `alpha`, which indicates the probability that the moving average of the 
true residuals falls within outputted interval. 

As an example the user wanted to track a standard row solver
 with a moving average of the sketched residuals of width 100 and get a 99% uncertainty sets
they could run the following code.  
```julia

using RLinearAlgebra

# Generate a system
A = rand(20, 5);
x = rand(5);
b = A * x;

# Specify solver
solver = RLSSolver(
    LinSysVecRowRandCyclic(),   # Random Cyclic Sampling
    LinSysVecRowProjStd(),      # Hyperplane projection
    LSLogFullMA(lambda_2 = 100),# Full Logger: maintains moving average residual history
    LSStopMaxIterations(200),   # Maximum iterations stopping criterion
    nothing                     # System solution (not solved yet)
);

# Solve the system
sol = rsolve(solver, A, b)
bounds = get_uncertainty(sol.log, alpha = .99)
```

**Note:
If the user is using a subset of identity type sampling method these uncertainty sets will perform conservatively
    because of a poor variance estimate. The user can reduce this conservativeness using the option $\eta=w$ in 
    the [`LSLogFullMA()`](@ref) settings for the solver, where w is a positive real number that divides the estimated variance of the set.**

## Stopping
In addition to being able to form the uncertainty sets, Pritchard and Patel also proposed a criterion for
stopping when using the sketched moving average estimator. `RLinearAlgebra.jl` allows for the specification 
these methods this can be done using [`LSStopMA()`](@ref). To understand the stopping criterion, it is valuable
to define some notation. If we allow $\rho_k^\lambda$ be the moving average of the true residuals, and 
$\hat \rho_k^\lambda$ be the moving average of the sketched residuals then two types of errors can occur.  

The first can be viewed as stopping too late, and it occurs when the tracking parameter value, 
$\rho_k^\lambda \leq \delta_I \upsilon$, while $\hat \rho_k^\lambda > \upsilon$, where $\delta_I$ 
is a user defined parameter that indicates a "meaningful gap" between $\hat \rho_k^\lambda$ 
and $\rho_k^\lambda$. 

The second error type can be viewed as stopping too early, and it occurs when the tracking parameter value, 
$\rho_k^\lambda \geq \delta_{II} \upsilon$, while $\hat \rho_k^\lambda < \upsilon$.
Where $\delta_{II}$ is a user defined parameter that indicates a "meaningful gap" between
$\hat \rho_k^\lambda$ and $\rho_k^\lambda$. 

Using `LSStopMA()` ensures that neither of these errors occurs
more than the user specified rates $\xi_{I}$ and $\xi_{II}$.
Where $\xi_{I}$ is the upper bound on the probability 
the solver is stopped too late and $\xi_{II}$ is an upper bound on
the probability that the solver is stopped too early.

The options for these stopping criterion parameters are represented by the options $\delta_I =$ `delta1`, $\delta_{II} = $ `delta2`,
$\xi_I =$ `chi1`, $\xi_{II}=$ `chi2`, and $\upsilon=$ `upsilon`. By default, `upsilon=1e-10`, `delta1=.9`, `delta2=1.1`, `chi_I = .01`,
`chi_{II} = .01`. To use the stopping criterion, the user must input a max number of iterations, and specify
changes to the default settings as options. So for instance if one wanted to use the stopping criterion with a 
maximum iteration of 1000 and `upsilon = 1e-3`, the following code could be used. 

```julia

using RLinearAlgebra

# Generate a system
A = rand(20, 5);
x = rand(5);
b = A * x;

# Specify solver
solver = RLSSolver(
    LinSysVecRowRandCyclic(),   # Random Cyclic Sampling
    LinSysVecRowProjStd(),      # Hyperplane projection
    LSLogFullMA(lambda_2 = 100),# Full Logger: maintains moving average residual history
    LSStopMA(1000, upsilon=1e-3),   # Maximum iterations stopping criterion
    nothing                     # System solution (not solved yet)
);

# Solve the system
sol = rsolve(solver, A, b)
bounds = get_uncertainty(sol.log, alpha = .99)
```
