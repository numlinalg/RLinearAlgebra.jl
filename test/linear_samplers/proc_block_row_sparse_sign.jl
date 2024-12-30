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

     # numsigns must between 0 and number of rows
     @test_throws AssertionError("`numsigns` must be \
          greater than 1.") LinSysBlkRowSparseSign(;numsigns = -5)
     @test_throws AssertionError("`numsigns` must be \
          greater than 1.") LinSysBlkRowSparseSign(;numsigns = 1)     
     SS = LinSysBlkRowSparseSign(numsigns = 15)
     @test_throws AssertionError("`numsigns` must less than the block size of \
          sketch matrix, 8.") RLinearAlgebra.sample(SS, A, b, x, 1)

     # numsigns' default value is correctly set, numsigns should equal to 8.
     SS = LinSysBlkRowSparseSign(block_size = 9)
     S, _, _ = RLinearAlgebra.sample(SS, A, b, x, 1)
     for i in axes(S, 2)
          @test length(S[:, i] == 0.0) == 1
     end

     # Block size must be positive
     @test_throws AssertionError("`block_size` must be \
          greater than 1.") LinSysBlkRowSparseSign(block_size = -12)
     @test_throws AssertionError("`block_size` must be \
          greater than 1.") LinSysBlkRowSparseSign(block_size = 1)

     # Block size less than matrix size test
     SS = LinSysBlkRowSparseSign(block_size = 12)
     @test_logs (:warn, "`block_size` should less than or \
          equal to row dimension, 10.") RLinearAlgebra.sample(SS, A, b, x, 1)


     SS = LinSysBlkRowSparseSign()
 
     v, M, res = RLinearAlgebra.sample(SS, A, b, x, 1)
     @assert SS.scaling == sqrt(size(A,1) / SS.numsigns)
     
     # Only test for res?
     for j = 2:5
          v, M, res = RLinearAlgebra.sample(SS, A, b, x, j)

          @test norm(M - v * A) < eps() * 1e2
          @test norm(res - (v * A * x - v * b)) < eps() * 1e2
          @assert SS.scaling == sqrt(size(A,1) / SS.numsigns)
     end
 
 end
 
 end
