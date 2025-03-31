module sparse_sign 
    using Test, RLinearAlgebra, Random
    include("../test_helpers/field_test_macros.jl")
    include("../test_helpers/approx_tol.jl")
    import SparseArrays: sparse, SparseMatrixCSC
    import LinearAlgebra: mul!, Adjoint
    using .FieldTest
    using .ApproxTol
    @testset "Compressor_Sparse_Sign" begin
        @test_compressor SparseSignRecipe
        # test the sparse spign constructor
        let
    	    Random.seed!(21321)
            # test the dimension checks in the construction of a sparse sign
            @test_throws DimensionMismatch SparseSign(Left, 2, 8)
            @test_throws ArgumentError SparseSign(Left, 2, -8)
            @test_throws ArgumentError SparseSign(Left, -2, 8)
    	    S1 = SparseSign()
    	    @test typeof(S1) <: Compressor
    	    @test S1.cardinality == Left
    	    @test S1.compression_dim == 8
    	    @test S1.nnz == 8
        end
        
        # Test the left compressor recipe construction and updating wirh default parameters
        let
    	    S = SparseSign()
    	    # Test completion of constructor
    	    A = rand(4, 2)
    	    b = rand(4)
    	    S_method = complete_compressor(S, A, b)
    	    @test typeof(S_method) <: CompressorRecipe
    	    # Check that default values are correct
    	    # and appropiate allocations have been made
    	    @test S_method.n_cols == size(A, 1)
    	    @test S_method.n_rows == 8
            # scale will be a tuple
            @test S_method.scale == [-1 / sqrt(8), 1 / sqrt(8)]
    	    @test typeof(S_method.op) <: SparseMatrixCSC
    	    # Test that update_compressor updates the entries
    	    signs = deepcopy(S_method.op.nzval)
    	    update_compressor!(S_method, zeros(4), A, b)
    	    # Check that at least one of the entries is different
    	    @test sum(signs .== S_method.op.nzval) < 4 * 8 
            # Check that the number of 
            # first collect the number of nonzeros per column
            n_f = sum([sum(S_method.op[:, i] .!= 0) for i = 1:4] .!= S_method.nnz)
            @test n_f  == 0 
        end

        # Test the right compressor recipe construction and updating wirh default parameters
        let
    	    S = SparseSign(cardinality = Right)
    	    # Test completion of constructor
    	    A = rand(2, 4)
    	    b = rand(4)
    	    S_method = complete_compressor(S, A, b)
    	    @test typeof(S_method) <: CompressorRecipe
    	    # Check that default values are correct
    	    # and appropiate allocations have been made
    	    @test S_method.n_cols == 8
    	    @test S_method.n_rows == size(A, 2)
            # scale will be a tuple
            @test S_method.scale == [-1 / sqrt(8), 1 / sqrt(8)]
            # RIght sketching should be an adjoint
    	    @test typeof(S_method.op) <: Adjoint 
    	    # Test that update_compressor updates the entries
    	    signs = deepcopy(S_method.op.parent.nzval)
    	    update_compressor!(S_method, zeros(4), A, b)
    	    # Check that at least one of the entries is different
    	    @test sum(signs .== S_method.op.parent.nzval) < 4 * 8 
            # Check that the number of 
            # first collect the number of nonzeros per rows
            n_f = sum([sum(S_method.op[i, :] .!= 0) for i = 1:4] .!= S_method.nnz)
            @test n_f  == 0 
    	end
    
        # Test the construction with keyword arguements changing dimensions
        let
    		Random.seed!(2131)
    	    n_rows = 4
    	    n_cols = 3
    	    A = rand(n_rows, n_cols)
    	    b = rand(n_rows)
    	    S1 = SparseSign(compression_dim = 2, nnz = 2)
    	    S_method = complete_compressor(S1, A, b)
    	    # Only test sizes here as all type issues should be caught in the first testset
    	    @test S_method.n_rows == 2
    	    @test S_method.n_cols == size(A, 1)
    	    @test S_method.scale == [-1 / sqrt(2), 1 / sqrt(2)]
    	    @test size(S_method) == (2, 4)

    	    S2 = SparseSign(cardinality = Right, compression_dim = 2, nnz = 2)
    	    S_method = complete_compressor(S2, A, b)
    	    # Only test sizes here as all type issues should be caught in the first testset
    	    @test S_method.n_cols == 2
    	    @test S_method.n_rows == size(A, 2)
    	    @test S_method.scale == [-1 / sqrt(2), 1 / sqrt(2)]
    	    @test size(S_method) == (3, 2)
        end
   
        # Teset the different multiplications with the left
        let
    		Random.seed!(2131)
    		n_rows = 20
    		n_cols = 3
    		nnz = 8
    		sketch_size = 10 
    	
    		A = rand(n_rows, n_cols)
    	    B = rand(sketch_size, n_cols)
    	    C1 = rand(sketch_size, n_cols)
            C1c = deepcopy(C1) 
    	    C2 = rand(n_rows, n_cols)
            C2c = deepcopy(C2) 
    	    x = rand(n_rows)
    	    y = rand(sketch_size)
            yc = deepcopy(y)
    		S_info = SparseSign(compression_dim = sketch_size)
    		S = complete_compressor(S_info, A)
    		
    		# Form a vector corresponding to the columns to be generate the sparse mat
    		nnz_cols = reduce(vcat, [repeat(i:i, S.nnz) for i = 1:n_rows])
            # Form the compressor to form the actual compressor
    		sparse_S = sparse(S.op.rowval, nnz_cols, S.op.nzval)
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
            # The unscaled versions are tested in the start multipliction
    	    mul!(C1, S, A, 2.0, 2.0)
    	    @test C1 ≈ 2.0 * sparse_S * A + 2.0 * C1c
    	    mul!(C2', B', S, 2.0, 2.0)
    	    @test C2' ≈ 2.0 * B' * sparse_S + 2.0 * C2c'
    	    mul!(y, S, x, 2.0, 2.0)
    	    @test y ≈ 2.0 * sparse_S * x + 2.0 * yc
            # make copy of x here because we have overwritten it above 
            xc = deepcopy(x)
    	    mul!(x, S', y, 2.0, 2.0)
    	    @test x ≈ 2.0 * sparse_S' * y + 2.0 * xc
        end

        # Teset the different multiplications with the right 
        let
    		Random.seed!(2131)
    		n = 20 
    		nnz = 8
    		sketch_size = 10 
    	
    		A = rand(n, sketch_size)
    	    B = rand(n, n)
    	    C1 = rand(sketch_size, sketch_size)
            C1c = deepcopy(C1)
            C2 = rand(n, sketch_size)
            C2c = deepcopy(C2)
    	    x = rand(sketch_size)
    	    y = rand(n)
            yc = deepcopy(y)
    		S_info = SparseSign(cardinality = Right, compression_dim = sketch_size)
    		S = complete_compressor(S_info, B)
    		
    		# Form a vector corresponding to the columns to be generate the sparse mat
    		nnz_cols = reduce(vcat, [repeat(i:i, S.nnz) for i = 1:n])
            # Form the compressor to form the actual compressor
    		sparse_S = sparse(S.op.parent.rowval, nnz_cols, S.op.parent.nzval)'
    	    # Test matrix multiplication from the left
    		@test S' * A ≈ sparse_S' * A
    	    # test transpose multiplication from the left 
    	    @test B * S ≈ B * sparse_S
            # Test multiplication from the right
            @test B' * S ≈ B' * sparse_S
            # Test transpose multiplication from the right
            @test A * S' ≈ A * sparse_S'
    	    # Test matrix vector multiplication from the left
    	    @test S * x ≈ sparse_S * x
    	    # Test multiplication from the right using transpose
    	    @test S' * y ≈ sparse_S' * y 
    	
    	    # Test the scalar addition portion of the multiplications
            # The unscaled versions are tested in the start multipliction
    	    mul!(C1, S', A, 2.0, 2.0)
    	    @test C1 ≈ 2.0 * sparse_S' * A + 2.0 * C1c
    	    mul!(C2, B, S, 2.0, 2.0)
    	    @test C2 ≈ 2.0 * B * sparse_S + 2.0 * C2c
    	    mul!(y, S, x, 2.0, 2.0)
    	    @test y ≈ 2.0 * sparse_S * x + 2.0 * yc
            # make copy of x here it was overwritten in previous mul 
            xc = deepcopy(x)
    	    mul!(x, S', y, 2.0, 2.0)
    	    @test x ≈ 2.0 * sparse_S' * y + 2.0 * xc
        end

    end

end
