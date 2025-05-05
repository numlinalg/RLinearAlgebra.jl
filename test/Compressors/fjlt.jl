module fjlt 
using Test, RLinearAlgebra, Random
import SparseArrays: sparse, SparseMatrixCSC, sprand
import LinearAlgebra: mul!, Adjoint
import Hadamard: hadamard
using ..FieldTest
using ..ApproxTol

Random.seed!(2131)
@testset "FJLT" begin
    @testset "FJLT: Compressor" begin
        # Verify Supertype
        @test supertype(FJLT) == Compressor

        # Verify fields and types
        @test fieldnames(FJLT) == (
            :cardinality, 
            :compression_dim, 
            :block_size, 
            :sparsity, 
            :type
        )
        @test fieldtypes(FJLT) == (Cardinality, Int64, Int64, Float64, Type{<:Number})

        # Verify the Internal Constructor
        let cardinality = Left(), comp_dim = 0, bs = 2, sparsity = 0.0, type = Float64
            @test_throws ArgumentError(
                "Field `compression_dim` must be positive."
            ) FJLT(
                cardinality, comp_dim, bs, sparsity, type
            )
        end

        let cardinality = Left(), comp_dim = 1, bs = 0, sparsity = 0.0, type = Float64
            @test_throws ArgumentError(
                "Field `block_size` must be positive."
            ) FJLT(
                cardinality, comp_dim, bs, sparsity, type
            )
        end

        let cardinality = Left(), comp_dim = 1, bs = 2, sparsity = 1.1, type = Float64
            @test_throws ArgumentError(
                "Field `sparsity` must be less than 1."
            ) FJLT(cardinality, comp_dim, bs, sparsity, type)
        end

        # Verify external constructor and type 
        for Card in [Left, Right]
            compressor = FJLT(; cardinality=Card())
            typeof(compressor.cardinality) == Card
        end

        for type in [Bool, Int16, Int32, Int64, Float16, Float32, Float64]
            compressor = FJLT(cardinality=Right(), type=type)
            @test compressor.type == type
        end

    end

    @testset "FJLT: CompressorRecipe" begin
        @test_compressor FJLTRecipe
        # test supertype
        @test supertype(FJLTRecipe) == CompressorRecipe
        @test fieldnames(FJLTRecipe) ==
            (:cardinality, :n_rows, :n_cols, :sparsity, :scale, :op, :signs, :padding)
        @test fieldtypes(FJLTRecipe) == (
            Cardinality,
            Int64,
            Int64,
            Float64,
            Float64,
            SparseMatrixCSC,
            BitVector,
            AbstractMatrix
        )

        # test the constructors
        let card = Left(),
            n_rows = 8,
            n_cols = 3,
            sparsity = .3,
            padded_dim = 8,
            comp_dim = 3,
            block_size = 10,
            type = Float16

            A = rand(n_rows, n_cols)
            recipe = FJLTRecipe(comp_dim, block_size, card, sparsity, A, type)
            @test typeof(recipe) == FJLTRecipe{
                typeof(card), 
                SparseMatrixCSC{type, Int64}, 
                Matrix{type}
            }
            @test recipe.cardinality == Left()
            @test recipe.n_rows == comp_dim
            @test recipe.n_cols == n_rows
            @test recipe.sparsity == sparsity
            @test recipe.scale == type(
                1 / (sqrt(padded_dim) * sqrt(comp_dim) * sqrt(sparsity))
            )
            @test typeof(recipe.op) <: SparseMatrixCSC
            @test eltype(recipe.op) == type
            @test size(recipe.op) == (comp_dim, padded_dim)
            @test typeof(recipe.signs) == BitVector
            @test size(recipe.padding) == (padded_dim, block_size) 
            @test eltype(recipe.padding) == type
        end

        # test the with sparsity of 0 and non power of 2 row 
        let card = Left(),
            n_rows = 6,
            n_cols = 3,
            sparsity = 0.0,
            padded_dim = 8,
            comp_dim = 3,
            block_size = 10,
            type = Float16

            A = rand(n_rows, n_cols)
            recipe = FJLTRecipe(comp_dim, block_size, card, sparsity, A, type)
            @test typeof(recipe) == FJLTRecipe{
                typeof(card), 
                SparseMatrixCSC{type, Int64}, 
                Matrix{type}
            }
            @test recipe.cardinality == Left()
            @test recipe.n_rows == comp_dim
            @test recipe.n_cols == n_rows
            @test recipe.sparsity == .25 * log(size(A,1))^2 / n_rows 
            sparsity = recipe.sparsity
            @test recipe.scale == type(
                1 / (sqrt(padded_dim) * sqrt(comp_dim) * sqrt(sparsity))
            )
            @test typeof(recipe.op) <: SparseMatrixCSC
            @test eltype(recipe.op) == type
            @test size(recipe.op) == (comp_dim, padded_dim)
            @test typeof(recipe.signs) == BitVector
            @test size(recipe.padding) == (padded_dim, block_size) 
            @test eltype(recipe.padding) == type
        end

        # test the with sparsity of 0 and non power of 2 row 
        let card = Right(),
            n_rows = 3,
            n_cols = 6,
            sparsity = 0.0,
            padded_dim = 8,
            comp_dim = 3,
            block_size = 10,
            type = Float16

            A = rand(n_rows, n_cols)
            recipe = FJLTRecipe(comp_dim, block_size, card, sparsity, A, type)
            @test typeof(recipe) == FJLTRecipe{
                typeof(card), 
                SparseMatrixCSC{type, Int64}, 
                Matrix{type}
            }
            @test recipe.cardinality == Right()
            @test recipe.n_rows == n_cols
            @test recipe.n_cols == comp_dim 
            @test recipe.sparsity == .25 * log(size(A, 2))^2 / n_cols
            sparsity = recipe.sparsity
            @test recipe.scale == type(
                1 / (sqrt(padded_dim) * sqrt(comp_dim) * sqrt(sparsity))
            )
            @test typeof(recipe.op) <: SparseMatrixCSC
            @test eltype(recipe.op) == type
            @test size(recipe.op) == (padded_dim, comp_dim)
            @test typeof(recipe.signs) == BitVector
            @test size(recipe.padding) == (block_size, padded_dim) 
            @test eltype(recipe.padding) == type
        end

    end

    @testset "FJLT: Complete Compressor" begin
        # test with left compressor
        let card = Left(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            sp = .3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                FJLT(; cardinality=card, compression_dim=c_dim, sparsity=sp, type=type), A
            )

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.n_cols == n_rows 
            @test compressor_recipe.sparsity == sp 
            @test compressor_recipe.scale == type(1 / (sqrt(4) * sqrt(.3) * sqrt(16)))
            @test typeof(compressor_recipe.op) <: SparseMatrixCSC
            @test typeof(compressor_recipe.signs) == BitVector
            @test typeof(compressor_recipe.padding) == Matrix{type}
        end

        # test the with sparsity of 0 and non power of 2 row 
        let card = Right(),
            n_rows = 3,
            n_cols = 6,
            sparsity = 0.3,
            padded_dim = 8,
            comp_dim = 3,
            block_size = 10,
            type = Float16

            A = rand(n_rows, n_cols)
            recipe = FJLTRecipe(comp_dim, block_size, card, sparsity, A, type)
            @test typeof(recipe) == FJLTRecipe{
                typeof(card), 
                SparseMatrixCSC{type, Int64}, 
                Matrix{type}
            }
            @test recipe.cardinality == Right()
            @test recipe.n_rows == n_cols
            @test recipe.n_cols == comp_dim 
            @test recipe.sparsity == sparsity 
            @test recipe.scale == type(
                1 / (sqrt(padded_dim) * sqrt(comp_dim) * sqrt(sparsity))
            )
            @test typeof(recipe.op) <: SparseMatrixCSC
            @test eltype(recipe.op) == type
            @test size(recipe.op) == (padded_dim, comp_dim)
            @test typeof(recipe.signs) == BitVector
            @test size(recipe.padding) == (block_size, padded_dim) 
            @test eltype(recipe.padding) == type
        end

        # test with right compressor
        let card = Right(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            sp = .3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                FJLT(; cardinality=card, compression_dim=c_dim, sparsity=sp, type=type), A
            )

            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == n_cols 
            @test compressor_recipe.n_cols == c_dim
            @test compressor_recipe.sparsity == sp 
            @test compressor_recipe.scale == type(1 / (sqrt(4) * sqrt(.3) * sqrt(16)))
            @test typeof(compressor_recipe.op) <: SparseMatrixCSC
            @test typeof(compressor_recipe.signs) == BitVector
            @test typeof(compressor_recipe.padding) == Matrix{type}
        end

    end

    @testset "FJLT: Update Compressor" begin
        # test with left compressor
        let card = Left(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            sp = .3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                FJLT(; cardinality=card, compression_dim=c_dim, sparsity=sp, type=type), A
            )

            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.op)
            oldsigns = deepcopy(compressor_recipe.signs)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == c_dim
            @test compressor_recipe.sparsity == sp 
            @test compressor_recipe.n_cols == 10 
            @test compressor_recipe.op != oldmat
            @test compressor_recipe.signs != oldsigns
        end

        # test with right compressor
        let card = Right(),
            n_rows = 10,
            n_cols = 10,
            c_dim = 4,
            sp = .3,
            type = Float16,
            A = rand(n_rows, n_cols),
            compressor_recipe = complete_compressor(
                FJLT(; cardinality=card, compression_dim=c_dim, sparsity=sp, type=type), A
            )

            # copy to test that the compressor has changed
            oldmat = deepcopy(compressor_recipe.op)
            oldsigns = deepcopy(compressor_recipe.signs)
            update_compressor!(compressor_recipe)
            # Test the values and types
            @test compressor_recipe.cardinality == card
            @test compressor_recipe.n_rows == 10 
            @test compressor_recipe.n_cols == c_dim
            @test compressor_recipe.sparsity == sp 
            @test compressor_recipe.op != oldmat
            @test compressor_recipe.signs != oldsigns
        end

    end

    # Test multimplcations with left compressors
    # Here we want to test the multiplication with matrices and vectors in the 
    # transposed and normal orientations for both the three and five argument mul!
    @testset "FJLT: Left Multiplication" begin
        # test the errors
        # C adn S have different rows
        let n_rows = 8,
            n_cols = 8,
            comp_dim = 2,
            A = rand(n_rows, n_cols),
            S = complete_compressor(FJLT(compression_dim = comp_dim), A),
            C = rand(comp_dim + 1, n_cols)
            @test_throws DimensionMismatch(
                "Matrix C has $(comp_dim + 1) rows while S has $comp_dim rows."
            ) mul!(C, S, A, 1.0, 0.0)
        end

        # C adn A have different columns 
        let n_rows = 8,
            n_cols = 8,
            comp_dim = 2,
            A = rand(n_rows, n_cols),
            S = complete_compressor(FJLT(compression_dim = comp_dim), A),
            C = rand(comp_dim, n_cols + 1)
            @test_throws DimensionMismatch(
                "Matrix C has $(n_cols+1) columns while A has $n_cols columns."
            ) mul!(C, S, A, 1.0, 0.0)
        end

        # A has more rows than S
        let n_rows = 8,
            n_cols = 8,
            comp_dim = 2,
            Agen = rand(n_rows, n_cols),
            A = rand(n_rows+1, n_cols),
            S = complete_compressor(FJLT(compression_dim = comp_dim), Agen),
            C = rand(comp_dim, n_cols)
            @test_throws DimensionMismatch(
                "Matrix A has more rows than the matrix S has columns."
            ) mul!(C, S, A, 1.0, 0.0)
        end

        # A has more columns than padding matrix in S
        let n_rows = 8,
            n_cols = 8,
            comp_dim = 2,
            Agen = rand(n_rows, n_cols),
            A = rand(n_rows, n_cols+1),
            S = complete_compressor(FJLT(compression_dim = comp_dim), Agen),
            C = rand(comp_dim, n_cols+1)
            @test_throws DimensionMismatch(
                "Matrix A has more columns than the padding matrix in S."
            ) mul!(C, S, A, 1.0, 0.0)
        end
        
        # Now test the five argument multiplication
        let n_rows = 20,
            n_cols = 3,
            c_dim = 10,
            pad_dim = 32,
            A = rand(n_rows, n_cols),
            B = rand(c_dim, n_cols),
            C1 = rand(c_dim, n_cols),
            C2 = rand(pad_dim, n_cols),
            x = rand(n_rows),
            y = rand(c_dim),
            pad_mat = zeros(pad_dim, n_cols)
            pad_mat2 = zeros(pad_dim, n_cols)

            # copies are for comparing with the "true version"
            C1c = deepcopy(C1)
            C2c = deepcopy(C2)
            yc = deepcopy(y)
            # Start by testing left sketching multiplications
            S_info = FJLT(compression_dim=c_dim)
            S = complete_compressor(S_info, A)

            mul!(C1, S, A, 1.0, 2.0)
            H = hadamard(pad_dim)
            pad_mat[1:n_rows, :] = A
            C_test = S.op * H * (ifelse.(S.signs, 1, -1) .* pad_mat) .* S.scale + C1c * 2.0
            @test C_test ≈ C1

            mul!(C2, S', B, 1.0, 2.0)
            fill!(padded_mat, 0.0)
            pad_mat[1:c_dim, :] = B
            C_test = S.op * H * (ifelse.(S.signs, 1, -1) .* pad_mat') .* S.scale + C2c * 2.0
            @test C_test ≈ C2
        end

    end

    # Test multimplcations with right compressors
    # Here we want to test the multiplication with matrices and vectors in the 
    # transposed and normal orientations for both the three and five argument mul!
    @testset "FJLT: Right Multiplication" begin
        let n = 20,
            nnz = 8,
            c_dim = 10,
            A = rand(n, c_dim),
            B = rand(n, n),
            C1 = rand(c_dim, c_dim),
            C2 = rand(n, c_dim),
            x = rand(c_dim),
            y = rand(n),

            C1c = deepcopy(C1)

            C2c = deepcopy(C2)
            yc = deepcopy(y)
            S_info = FJLT(; cardinality=Right(), compression_dim=c_dim)
            S = complete_compressor(S_info, B)
            # Form a vector corresponding to the columns to be generate the sparse mat
            nnz_cols = reduce(vcat, [repeat(i:i, S.nnz) for i in 1:n])
            # Form the compressor to form the actual compressor
            sparse_S = sparse(S.op.parent.rowval, nnz_cols, S.op.parent.nzval)'

            # Using * will test three element mul and * multiplication
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
            # Test multiplying the matrix to a vector from the right
            @test y' * S ≈ y' * sparse_S
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

end
