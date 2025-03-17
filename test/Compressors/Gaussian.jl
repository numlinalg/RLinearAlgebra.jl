module Gaussian_compressor
    using Test, RLinearAlgebra, Random, LinearAlgebra
    import Base.:*
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
            @test S1.n_rows == 0
            @test S1.n_cols == 0
            # Test completion of constructor
            A = rand(4, 2)
            b = rand(4)
            S_method = complete_compressor(S1, A)
            @test typeof(S_method) <: CompressorRecipe
        end

        let
            Random.seed!(21321)
            S1 = Gaussian()
            @test typeof(S1) <: Compressor
            @test S1.n_rows == 0
            @test S1.n_cols == 0
            # Test completion of constructor
            A = rand(4, 2)
            b = rand(4)
            S_method = complete_compressor(S1, A)
            @test typeof(S_method) <: CompressorRecipe
            # Check that default values are correct and appropriate allocations have been made
            @test S_method.n_cols == size(A, 1)
            @test S_method.n_rows == 2
            sample_size = min(S_method.n_rows, S_method.n_cols)
            @test S_method.scale == 1 / sqrt(sample_size)
            @test typeof(S_method.sketch_matrix) <: Matrix{Float64}
            # Test that update_compressor updates the entries
            sketch_matrix = deepcopy(S_method.sketch_matrix)
            update_compressor!(S_method)
            # Check that at least one of the entries is different
            @test sum(sketch_matrix .== S_method.sketch_matrix) < 4 * 2 
        end

        let
            Random.seed!(2131)
            n_rows = 4
            n_cols = 3
            A = rand(n_rows, n_cols)
            b = rand(n_rows)
            # Test the case where two dimensions are input but do not match with matrix A
            S3 = Gaussian(n_rows = 2, n_cols = 7)
            @test_throws AssertionError("Either you inputted row or column dimension must match \\
                the column or row dimension of the matrix.") complete_compressor(S3, A)
        end

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
            S_info = Gaussian(n_rows = sketch_size)
            S = complete_compressor(S_info, A)
            
            # Form a vector corresponding to the columns to generate the sketch matrix
            S_test = S.sketch_matrix
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

    end

end