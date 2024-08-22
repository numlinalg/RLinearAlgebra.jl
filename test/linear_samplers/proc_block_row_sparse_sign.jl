# This file is part of RLinearAlgebra.jl
# This file was written by Tunan Wang

module ProceduralTestLSBRSparseSign

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSBR Sparse Sign -- Procedural" begin

     # Verify appropriate super type
     @test supertype(LinSysBlkRowSparseSign) == LinSysBlkRowSampler
 
     # Test whether row ordering remains fixed
     A = rand(10,6)
     b = rand(10)
     x = rand(6)
 

     # Test assertions

     # Sparsity must between 0 and 1
     SS = LinSysBlkRowSparseSign(sparsity = -5)
     @test_throws AssertionError("`sparsity` must be in the range of (0, 1)") RLinearAlgebra.sample(SS, A, b, x, 1)
     SS = LinSysBlkRowSparseSign(sparsity = 5)
     @test_throws AssertionError("`sparsity` must be in the range of (0, 1)") RLinearAlgebra.sample(SS, A, b, x, 1)

     # Block size must be positive
     SS = LinSysBlkRowSparseSign(block_size = -12)
     @test_throws AssertionError("`block_size` must be positive") RLinearAlgebra.sample(SS, A, b, x, 1)

     # Block size less than matrix size test
     SS = LinSysBlkRowSparseSign(block_size = 12)
     @test_throws AssertionError("`block_size` must be less than row dimension") RLinearAlgebra.sample(SS, A, b, x, 1)


     SS = LinSysBlkRowSparseSign()
 
     v, M, res = RLinearAlgebra.sample(SS, A, b, x, 1)
     
     # Only test for res?
     for j = 2:5
          v, M, res = RLinearAlgebra.sample(SS, A, b, x, j)
          @test norm(res - (v * A * x - v * b)) < eps() * 1e2
     end
 
 end
 
 end
