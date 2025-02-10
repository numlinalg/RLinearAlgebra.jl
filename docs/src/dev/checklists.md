# Development Checklists 

```@contents
Pages=["checklists.md"]
```
The purpose of this page is to maintain checklists of tasks to complete when adding new 
methods to the library. These checklists are organized by method.

## Compressors 
If you are implementing a compression method for the library, make sure you have completed 
the following steps before making a pull request. 

```
1. Implementation
- [ ] Create a file in the directory `Compressors`.
- [ ] Create a Compressor structure with n_rows and n_cols as well as other user-controlled 
parameters.
- [ ] Create a constructor with keyword default values for your struct.
- [ ] Create a CompressorRecipe structure that uses the parameters from the Compressor 
structure to preallocate memory.
- [ ] Create a `complete_compressor` function that takes as inputs of the Compressor, A, x,b
and returns a CompressorRecipe
- [ ] Create a `update_compressor!` function that generates new random values for a random
components of the Compressor 
- [ ] a 5 input `mul!` function for  applying a compressor to a matrix from the left
- [ ] a 5 input `mul!` function for  applying a compressor to a matrix from the right 
- [ ] a 5 input `mul!` function for  applying a compressor to a vector
- [ ] a 5 input `mul!` function for  applying the adjoint of the compressor to a vector
- [ ] Add an include("Compressors/[YOURFILE]") at bottom of page 
- [ ] Add import statements to src/RLinearAlgebra.jl with any functions from other packages 
that you use.
- [ ] Add your Compressor and CompressorRecipe to src/RLinearAlgebra.jl.
- [ ] Add your Compressor and CompressorRecipe to docs/src/api/compressors.md 
under the appropiate heading. 
- [ ] Add a procedural test to test/linear_samplers. Be sure to check that the functions 
work as intended and all warnings/assertions are displayed.
2. Pull request
- [ ] Give a specific title to pull request.
- [ ] Lay out the features added to the pull request.
- [ ] Tag two people to review your pull request.
```

