module ProceduralFWHT
    using RLinearAlgebra, Hadamard, Random, Test, LinearAlgebra

    @testset "Fast Walsh Hadamard Transform" begin
        Random.seed!(1010)
        
        # Test vector of size 9 for error
        x = rand(9)
        @test_throws AssertionError("Size of vector must be power of 2.") RLinearAlgebra.fwht!(x)

        
        # Generate test vector of size 8
        x = rand(8)
        xc = deepcopy(x)
        # Scaling Factor
        sc = 1/sqrt(8)
        # Bit Vector
        sgn = bitrand(8)
        # Signs corresponding to bit vector
        signs = [sgn[i] ? 1 : -1 for i in 1:8]
        # Hadamard Matrix
        H = hadamard(8)
    
        # Test Basic transform
        RLinearAlgebra.fwht!(x)
        @test norm(x - H * xc) < 1e-10
        
        # Test Transform with scaling
        x = deepcopy(xc)
        RLinearAlgebra.fwht!(x, scaling = sc)
        @test  norm(x - H * (xc .* sc)) < 1e-10
        # Test that applying this again gives original vector
        RLinearAlgebra.fwht!(x, scaling = sc)
        @test  norm(x - xc) < 1e-10

        # Test with the sign flipping
        x = deepcopy(xc)
        RLinearAlgebra.fwht!(x, signs = sgn)
        @test  norm(x - H * (xc .* signs)) < 1e-10

        # Test with the sign flipping and scaling
        x = deepcopy(xc)
        RLinearAlgebra.fwht!(x, signs = sgn, scaling = sc)
        @test  norm(x - H * ((xc .* signs) .* sc)) < 1e-10
    end

end