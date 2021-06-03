using Test
using RLinearAlgebra

#verbose = true, Julia 1.6
@testset "Linear System Sampler Tests" begin
      for (testset_name, tests) in RLinearAlgebra.linear_samplers_testset_proc
            @testset "$testset_name" begin
                  [eval(tst) for tst in tests]
            end
      end
end
