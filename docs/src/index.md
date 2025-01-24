# RLinearAlgebra

[RLinearAlgebra](https://github.com/numlinalg/RLinearAlgebra.jl) is a Julia library for the 
development and application of randomized algorithms to the problems of forming low rank 
approximations to matrices and finding the solution of linear systems of equations. 
Because of the large diversity of randomized techniques, rather than offering isolated 
routine implementations of algorithms, this library implements a series of extendable data 
structures and methods which allow code reuse.



To facilitate benchmarking, we provide an abstract class for solver solution logging and stopping criteria.

## Documentation structure

This documentation serves both as a manual to the library and as an introduction to 
randomized linear approximation techniques and randomized linear algebra solvers. 
We divide it in four parts:

* **Tutorial**: here we offer examples of how to solve linear systems with RLinearAlgebra and how to extend the library.
* **Manual**: here we offer an introduction to solving linear systems with randomized linear 
algebra techniques. We introduce theoretical foundations and we provide code examples with RLinearAlgebra.
* **API**: a detailed description of all the classes and methods of the library.
* **Development**: detailed instructions on how to contribute to the library.
## Acknowledgements
This work is supported by the National Science Foundation Office of Advanced Cyberinfrastructure under awards [2309445](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2309445) and [2309446](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2309446).
This material is based upon work supported by the U.S. Department of Energy, Office of Science, under contract number DE-AC02-06CH11357.
