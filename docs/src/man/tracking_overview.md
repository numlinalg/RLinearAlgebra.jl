# Tracking an Iterative Sketching Solver

When using an iterative sketching method like Randomized Block Kaczmarz
known the quality of the solution is integral for being able to make
stopping decisions. 
it is important to track these techniques in a manner that does 
not undermine the stated benefits of the method, namely being able 
update a solution using only a row or block of rows of the matrix.
Na\"ive methods of tracking like computing the norm of the residual,
$\|Ax-b\|_2$, directly undermine these benefits, by requiring access
to the full matrix to compute. In fact, when the row dimension of the 
underlying linear system is far greater than block size, computing 
the progress estimator can be substantially more expensive than computing
the update itself. The RLinearAlgebra library implemented a technique from 
  
