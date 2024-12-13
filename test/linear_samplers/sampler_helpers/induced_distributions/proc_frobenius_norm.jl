# Date: 12/13/2024
# Author: Christian Varner
# Purpose: test the implementation for creating a distribution using the
# forbenius norm of a matrix

module ProceduralFrobeniusNormDistribution

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "Distribution by Frobenius Norm -- Procedural" begin

    Random.seed!(1010)

    # test definition
    @test isdefined(RLinearAlgebra, :frobenius_norm_distribution)

    # test return value is correct size and indeed sums to 1
    dims = [(100, 50), (50, 100), (100, 100)]
    row_distribution = [true, false]
    for (nrow, ncol) in dims
        for dist_type in row_distribution
            A = randn(nrow, ncol)
            distribution = RLinearAlgebra.frobenius_norm_distribution(A, dist_type)

            @test typeof(distribution) == Vector{Float64}
            
            sz = dist_type ? size(A, 1) : size(A, 2)  
            @test length(distribution) == sz
            @test sum(distribution) ≈ 1 atol = 1e-14

            max_index = dist_type ? nrow : ncol
            for i in 1:max_index
                value = dist_type ? norm(@view A[i, :])^2 / norm(A)^2 : norm(@view A[:, i])^2 / norm(A)^2  
                @test distribution[i] ≈ value
            end 
        end
    end
end

end