module sparse_sign 
    using Test, RLinearAlgebra, Random
    include("../test_helpers/field_test_macros.jl")
    include("../test_helpers/approx_tol.jl")
    import SparseArrays: sparse, SparseMatrixCSC
    import LinearAlgebra: mul!
    using .FieldTest
    using .ApproxTol
    @testset "Compressor_Sparse_Sign" begin
        @test_compressor SparseSignRecipe
        let
    	    Random.seed!(21321)
    	    S1 = SparseSign()
    	    @test typeof(S1) <: Compressor
    	    @test S1.n_rows == 0
    	    @test S1.n_cols == 0
    	    @test S1.nnz == 8
    	    # Test completion of constructor
    	    A = rand(4, 2)
    	    b = rand(4)
    	    S_method = complete_compressor(S1, A, b)
    	    @test typeof(S_method) <: CompressorRecipe
    	    # Check that default values are correct
    	    # and appropiate allocations have been made
    	    @test S_method.n_cols == size(A, 1)
    	    @test S_method.n_rows == 2
    	    @test S_method.scale == 1 / sqrt(2)
    	    @test typeof(S_method.Mat) <: SparseMatrixCSC
    	    # Test that update_compressor updates the entries
    	    signs = deepcopy(S_method.Mat.nzval)
    	    update_compressor!(S_method, A, b, zeros(4))
    	    # Check that at least one of the entries is different
    	    @test sum(signs .== S_method.Mat.nzval) < 4 * 2 
    	end
    
        let
    	    # Begin by checking the errors that occur when nnz > compression
    		Random.seed!(2131)
    	    n_rows = 4
    	    n_cols = 3
    	    S1 = SparseSign(nnz = 9)
    	    A = rand(n_rows, n_cols)
    	    b = rand(n_rows)
    	    @test_throws AssertionError S_method = complete_compressor(S1, A, b)
    	    S2 = SparseSign(n_cols = 2)
    	    S_method = complete_compressor(S2, A, b)
    	    # Only test sizes here as all type issues should be caught in the first testset
    	    @test S_method.n_cols == 2
    	    @test S_method.n_rows == size(A, 2)
    	    @test S_method.scale == 1 / sqrt(2)
    	    @test size(S_method) == (n_cols, 2)
    	    S2 = SparseSign(n_rows = 3)
    	    S_method = complete_compressor(S2, A, b)
    	    # Only test sizes here as all type issues should be caught in the first testset
    	    @test S_method.n_cols == size(A, 1) 
    	    @test S_method.n_rows == 3 
    	    @test S_method.scale == 1 / sqrt(3)
    	    @test size(S_method) == (3, 4)
            # Test the case where two dimensions are input but do not match the 
            S3 = SparseSign(n_rows = 2, n_cols = 7)
            @test_throws AssertionError complete_compressor(S3, A, b)
        end
    
        let
    		Random.seed!(2131)
    		n_rows = 20
    		n_cols = 3
    		nnz = 8
    		sketch_size = 10 
    	
    		A = rand(n_rows, n_cols)
    	    B = rand(sketch_size, n_cols)
    	    C1 = rand(sketch_size, n_cols)
    	    C2 = rand(n_rows, sketch_size)
    	    C3 = rand(n_rows, n_cols)
    	    x = rand(n_rows)
    	    y = rand(sketch_size)
    		S_info = SparseSign(n_rows = sketch_size)
    		S = complete_compressor(S_info, A)
    		
    		# Form a vector corresponding to the columns to be generate the sparse mat
    		nnz_cols = reduce(vcat, [repeat(i:i, S.nnz) for i = 1:n_rows])
    		sparse_S = sparse(S.Mat.rowval, nnz_cols, S.Mat.nzval)
    	    # Test matrix multiplication from the left
    		@test S * A ≈ sparse_S * A
    	    # test transpose multiplication from the left 
    	    @test S' * B ≈ sparse_S' * B
            # Test multiplication from the right
            @test B' * S ≈ B' * sparse_S
            # Test transpose multiplication from the right
            @test A' * S' ≈ A' * sparse_S'
    	    # Test matrix vector multiplication from the left
    	    @test S * x ≈ sparse_S * x
    	    # Test multiplication from the right using transpose
    	    @test S' * y ≈ sparse_S' * y 
    	
    	    # Test the scalar addition portion of the multiplications
    	    mul!(C1, S, A, 2.0, 0.0)
    	    @test C1 ≈ 2.0 * sparse_S * A
    	    mul!(C3', B', S, 2.0, 0.0)
    	    @test C3' ≈ 2.0 * B' * sparse_S 
    	    mul!(y, S, x, 2.0, 0.0)
    	    @test y ≈ 2.0 * sparse_S * x 
    	    mul!(x, S', y, 2.0, 0.0)
    	    @test x ≈ 2.0 * sparse_S' * y 
        end

    end

end
