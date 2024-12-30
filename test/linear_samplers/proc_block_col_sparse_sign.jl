# This file is part of RLinearAlgebra.jl
# This file was written by Tunan Wang

module ProceduralTestLSBRSparseSign

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSBC Sparse Sign -- Procedural" begin

     # Verify appropriate super type
     @test supertype(LinSysBlkColSparseSign) == LinSysBlkColSampler
 
     # Test construction
     A = rand(5,10)
     b = rand(5)
     x = rand(10)

 
     # Test assertions and default values

     # numsigns must between 2 and number of rows, block_size is set to 8 in default.
     @test_throws AssertionError("`numsigns` must be \
          greater than 1.") LinSysBlkColSparseSign(;numsigns = -5)
     @test_throws AssertionError("`numsigns` must be \
          greater than 1.") LinSysBlkColSparseSign(;numsigns = 1)
     SS = LinSysBlkColSparseSign(;numsigns = 15)
     @test_throws AssertionError("`numsigns` must less than the block size of \
          sketch matrix, 8.") RLinearAlgebra.sample(SS, A, b, x, 1)

     # numsigns' default value is correctly set, numsigns should equal to 8.
     SS = LinSysBlkColSparseSign(block_size = 9)
     S, _, _, _ = RLinearAlgebra.sample(SS, A, b, x, 1)
     for i in axes(S, 1)
          @test length(S[i, :] == 0.0) == 1
     end

     # Block size must be greater than 1
     @test_throws AssertionError("`block_size` must be \
          greater than 1.") LinSysBlkColSparseSign(block_size = -12)
     @test_throws AssertionError("`block_size` must be \
          greater than 1.") LinSysBlkColSparseSign(block_size = 1)

     # Block size less than matrix size test
     SS = LinSysBlkColSparseSign(block_size = 12)
     @test_logs (:warn, "`block_size` should less than or \
          equal to column dimension, 10.") RLinearAlgebra.sample(SS, A, b, x, 1)


     SS = LinSysBlkColSparseSign()
 
     for i in 1:5
          S, AS, grad, res = RLinearAlgebra.sample(SS, A, b, x, i)
  
          @test norm(AS - A * S) < eps() * 1e2
          @test norm(res - (A * x - b)) < eps() * 1e2
          @test norm(grad - S' * A' * (A * x - b)) < eps() * 1e2
          @assert SS.scaling == sqrt(size(A,2) / SS.numsigns)
     end
 
 end
 
 end
