Most Randomized Linear Algebra routines work by forming a low-dimensional representation of 
a matrix or vector. These representations are typically formed by multiplying some form of 
random matrix with the matrix or vector we want to represent in a low-dimensional space. 
For instance, if we have ``x \\in \\mathbb{R}^{10,000}`` we can apply a matrix, 
``S \\in \\mathbb{R}^{20 \\times 10,000}`` made up of 
independent and identically distributed ``\\textbf{Normal}(0, 1/\\sqrt{20})`` to obtain
``y = Sx \\in \\mathbb{R}^{20}`` where
``(1-\\epsilon)\|x\| \\leq \|y\| \\leq (1+\epsilon) \|x\|`` with high probability. 

Of course, many other techniques beyond the one described above can be used to generate 
`S` and they vary both in terms of their approximation capabilities and the speed they can 
be applied to a matrix. In papers, Randomized Linear Algebraists often refer to techniques
for generating `S` as either sampling (random subset of identity) or sketching 
(general random matrix) techniques. For simplicity RLinearAlgebra.jl refers to both types 
of techniques under the general family of Compressors.

