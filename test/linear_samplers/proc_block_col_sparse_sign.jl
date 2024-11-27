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

 
     # Test assertions

     # numsigns must between 0 and number of rows
     SS = LinSysBlkColSparseSign(numsigns = -5)
     @test_throws AssertionError("`numsigns` Must be strictly between 0 and 10") RLinearAlgebra.sample(SS, A, b, x, 1)
     SS = LinSysBlkColSparseSign(numsigns = 15)
     @test_throws AssertionError("`numsigns` Must be strictly between 0 and 10") RLinearAlgebra.sample(SS, A, b, x, 1)
     SS = LinSysBlkColSparseSign(numsigns = 5)
     @test RLinearAlgebra.sample(SS, A, b, x, 1) !== nothing

     # Block size must be positive
     SS = LinSysBlkColSparseSign(block_size = -12)
     @test_throws AssertionError("`block_size` must be positive") RLinearAlgebra.sample(SS, A, b, x, 1)

     # Block size less than matrix size test
     SS = LinSysBlkColSparseSign(block_size = 12)
     @test_logs (:warn, "`block_size` should be less than or equal to column dimension") RLinearAlgebra.sample(SS, A, b, x, 1)
     # @test_throws AssertionError("`block_size` must be less than row dimension") RLinearAlgebra.sample(SS, A, b, x, 1)

     # Block size is correct
     SS = LinSysBlkColSparseSign(block_size = 5)
     @test RLinearAlgebra.sample(SS, A, b, x, 1) !== nothing


     SS = LinSysBlkColSparseSign()
 
     for i in 1:5
          S, AS, grad, res  = RLinearAlgebra.sample(SS, A, b, x, i)
  
          @test norm(AS - A * S) < eps() * 1e2
          @test norm(res - (A * x - b)) < eps() * 1e2
          @test norm(grad - S' * A' * (A * x - b)) < eps() * 1e2
     end
 
 end
 
 end
