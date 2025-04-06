module Gaussian_compressor
    using Test, RLinearAlgebra, Random
    import Base.:*
    import LinearAlgebra: transpose, adjoint
    import LinearAlgebra: mul!, lmul!
    include("../../src/Compressors.jl")
    include("../../src/Compressors/Gaussian.jl")
    include("../../src/RLinearAlgebra.jl")
    include("../test_helpers/field_test_macros.jl")
    include("../test_helpers/approx_tol.jl")
    using .FieldTest
    using .ApproxTol

    @testset "Compressor_Gaussian" begin
        @test_compressor GaussianRecipe
        let 
            Random.seed!(21321)
            S1 = Gaussian()
            @test typeof(S1) <: Compressor
            @test S1.cardinality == Left
            @test S1.compression_dim == 2
            # Test the case when the compression_dim input is not positive
            @test_throws ArgumentError Gaussian(cardinality = Left, compression_dim = -7)
        end
        
        # Test the left compressor recipe construction and updating wirh default parameters
        let
            Random.seed!(21321)
            S1 = Gaussian()
            @test typeof(S1) <: Compressor
            @test S1.cardinality == Left
            @test S1.compression_dim == 2
            # Test completion of constructor
            A = rand(4, 2)
            S_method = complete_compressor(S1, A)
            @test typeof(S_method) <: CompressorRecipe
            # Check that default values are correct and appropriate allocations have been made
            @test S_method.cardinality == Left
            @test S_method.compression_dim == 2
            @test S_method.n_rows == S1.compression_dim
            @test S_method.n_cols == size(A,1)
            sample_size = S_method.compression_dim
            @test S_method.scale == 1 / sqrt(sample_size)
            @test typeof(S_method.op) <: Matrix{eltype(A)}
            @test size(S_method.op, 2) == size(A, 1)
            # Test that update_compressor updates the entries
            sketch_matrix = deepcopy(S_method.op)
            update_compressor!(S_method)
            # Check that at least one of the entries is different
            @test sum(sketch_matrix .== S_method.op) < 4 * 2 
        end

        # Test the right compressor recipe construction and updating wirh default parameters
        let
            Random.seed!(21321)
            S2 = Gaussian(cardinality = Right)
            @test typeof(S2) <: Compressor
            @test S2.cardinality == Right
            @test S2.compression_dim == 2
            # Test completion of constructor
            A = rand(2, 6)
            S_method = complete_compressor(S2, A)
            @test typeof(S_method) <: CompressorRecipe
            # Check that default values are correct and appropriate allocations have been made
            @test S_method.cardinality == Right
            @test S_method.compression_dim == 2
            @test S_method.n_rows == size(A,2)
            @test S_method.n_cols == S2.compression_dim
            sample_size = S_method.compression_dim
            @test S_method.scale == 1 / sqrt(sample_size)
            @test typeof(S_method.op) <: Matrix{eltype(A)}
            @test size(S_method.op, 1) == size(A, 2)
            # Test that update_compressor updates the entries
            sketch_matrix = deepcopy(S_method.op)
            update_compressor!(S_method)
            # Check that at least one of the entries is different
            @test sum(sketch_matrix .== S_method.op) < 6 * 2 
        end

        # Test the different multiplications with the left
        let
            Random.seed!(2131)
            n_rows = 10
            n_cols = 3
            sketch_size = 6
            A = rand(n_rows, n_cols)
            B = rand(sketch_size, n_cols)
            C1 = rand(sketch_size, n_cols)
            C2 = rand(n_rows, sketch_size)
            C3 = rand(n_rows, n_cols)
            x = rand(n_rows)
            y = rand(sketch_size)
            S_info = Gaussian(cardinality = Left, compression_dim = sketch_size)
            S = complete_compressor(S_info, A)
            
            # Form a vector corresponding to the columns to generate the sketch matrix
            S_test = S.op
            # Test matrix multiplication from the left
            @test S * A ≈ S_test * A
            # Using transpose will test matrix multiplication from right
            @test S' * B ≈ S_test' * B
            # Test matrix vector multiplication from the left
            @test S * x ≈ S_test * x
            # Test multiplication from the right using transpose
            @test S' * y ≈ S_test' * y

            # Test the scalar addition portion of the multiplications
            mul!(C1, S, A, 2.0, 0.0)
            @test C1 ≈ 2.0 * S_test * A
            mul!(C3', B', S, 2.0, 0.0)
            @test C3' ≈ 2.0 * B' * S_test 
            mul!(y, S, x, 2.0, 0.0)
            @test y ≈ 2.0 * S_test * x 
            mul!(x, S', y, 2.0, 0.0)
            @test x ≈ 2.0 * S_test' * y 
        end

        # Test the different multiplications with the right
        let
            Random.seed!(2131)
            n_rows = 3
            n_cols = 10
            sketch_size = 6
            A = rand(n_rows, n_cols)
            B = rand(n_cols, sketch_size)
            C1 = rand(n_rows, sketch_size)
            C3 = rand(sketch_size, sketch_size)
            x = rand(sketch_size)
            y = rand(n_cols)
            S_info = Gaussian(cardinality = Right, compression_dim = sketch_size)
            S = complete_compressor(S_info, A)
            
            # Form a vector corresponding to the columns to generate the sketch matrix
            S_test = S.op
            # Test matrix multiplication from the right
            @test A * S ≈ A * S_test
            # Test transpose from right
            @test B * S' ≈ B * S_test'
            # Test matrix multiplication from the left
            @test S * B' ≈ S_test * B'
            # Using transpose from left
            @test S' * A' ≈ S_test' * A'
            # Test matrix vector multiplication from the left
            @test S * x ≈ S_test * x
            # Test multiplication from the right using transpose
            @test S' * y ≈ S_test' * y

            # Test the scalar addition portion of the multiplications
            mul!(C1, A, S, 2.0, 0.0)
            @test C1 ≈ 2.0 * A * S_test
            mul!(C3, S', B, 2.0, 0.0)
            @test C3' ≈ 2.0 * B' * S_test 
            mul!(y, S, x, 2.0, 0.0)
            @test y ≈ 2.0 * S_test * x 
            mul!(x, S', y, 2.0, 0.0)
            @test x ≈ 2.0 * S_test' * y 
        end

    end

end