# This file is part of RLinearAlgebra.jl

module ProceduralTestGent

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "Gent Projection -- Procedural" begin
    Random.seed!(12312)
    A = rand(10,5)
    x = rand(5)
    b = A * x
    # Fixing block size to 3
    G = RLinearAlgebra.GentData(A, 3)
    # Check sizes of block matrix B 
    m, n = size(G.B)
    # m should  total columns in matrix plus rows in a block plus one for the residual
    @test m == 3 + 1 + 5
    # n should be number of columns blus one for constant vector
    @test n == 5 + 1

    #Test block copy function 
    # Copying rows 1:3 of the matrix and testing the appropiate location
    RLinearAlgebra.copy_block_from_mat!(G.B, A, b, 1:3)
    @test A[1:3, :] == G.B[7:9, 1:5] 
    @test b[1:3] == G.B[7:9, 6]
    #Test for when less than 3 rows are copied
    RLinearAlgebra.copy_block_from_mat!(G.B, A, b, 1:2)
    @test A[1:3, :] == G.B[7:9, 1:5] 
    @test b[1:3] == G.B[7:9, 6]
    #Test for when less than 3 rows are copied
    @test_throws AssertionError("The block indices must be less than `block_size`.") RLinearAlgebra.copy_block_from_mat!(G.B, A, b, 1:4)
    @test A[1:3, :] == G.B[7:9, 1:5] 
    @test b[1:3] == G.B[7:9, 6]
    # Test everything below upper triangular portion
    # is zero
    RLinearAlgebra.gentleman!(G)
    for i in 1:n
        for j in (i+1):m
            @test G.B[j,i] == 0
        end

    end

    # Test ldiv!
    y = zeros(5)
    ldiv!(y, G, b)
    @test norm(y - x) < 1e2 * eps()

    # Test Reset gent makes everything zero 
    RLinearAlgebra.reset_gent!(G)
    for i in 1:n  
        for j in 1:m
            @test G.B[j,i] == 0
        end

        @test G.B[i] == 0
    end

end

end # End module
