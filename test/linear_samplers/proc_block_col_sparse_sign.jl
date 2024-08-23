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

     # Sparsity must between 0 and 1
     SS = LinSysBlkColSparseSign(sparsity = -5)
     @test_throws AssertionError("`sparsity` must be in the range of (0, 1)") RLinearAlgebra.sample(SS, A, b, x, 1)
     SS = LinSysBlkColSparseSign(sparsity = 5)
     @test_throws AssertionError("`sparsity` must be in the range of (0, 1)") RLinearAlgebra.sample(SS, A, b, x, 1)

     # Block size must be positive
     SS = LinSysBlkColSparseSign(block_size = -12)
     @test_throws AssertionError("`block_size` must be positive") RLinearAlgebra.sample(SS, A, b, x, 1)

     # Block size less than matrix size test
     SS = LinSysBlkColSparseSign(block_size = 12)
     @test_logs (:warn, "`block_size` shoould be less than column dimension.") RLinearAlgebra.sample(SS, A, b, x, 1)


     SS = LinSysBlkColSparseSign()
 
     for i in 1:5
          S, AS, grad, res  = RLinearAlgebra.sample(SS, A, b, x, i)
  
          @test norm(AS - A * S) < eps() * 1e2
          @test norm(res - (A * x - b)) < eps() * 1e2
          @test norm(grad - S' * A' * (A * x - b)) < eps() * 1e2
     end
 
 end
 
 end
