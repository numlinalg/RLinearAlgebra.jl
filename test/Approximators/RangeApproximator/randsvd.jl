module  RandomizedSVD
using Test, RLinearAlgebra, LinearAlgebra
import RLinearAlgebra: complete_compressor
import LinearAlgebra: mul!, svd, Diagonal
using ..FieldTest
using ..ApproxTol

mutable struct TestCompressor <: Compressor
    cardinality::Cardinality
    compression_dim::Int64
end

TestCompressor() = TestCompressor(Right(), 5)

mutable struct TestCompressorRecipe <: CompressorRecipe 
    cardinality::Cardinality
    n_rows::Int64
    n_cols::Int64
    op::AbstractMatrix
end

function RLinearAlgebra.complete_compressor(comp::TestCompressor, A::AbstractMatrix)
    n_cols = comp.compression_dim
    n_rows = size(A, 2)
    # Make a gaussian compressor
    op = randn(n_rows, n_cols) ./ sqrt(n_cols)
    return TestCompressorRecipe(comp.cardinality, n_rows, n_cols, op)
end

# Define a mul function for the test compressor
function RLinearAlgebra.mul!(
    C::AbstractMatrix, 
    A::AbstractMatrix, 
    S::Main.RandomizedSVD.TestCompressorRecipe, 
    alpha::Float64, 
    beta::Float64
)
    mul!(C, A, S.op, alpha, beta)
end

@testset "RandSVD" begin 
    @testset "RandSVD" begin
        supertype(RandSVD) == Approximator

        # test the fieldnames and types
        fieldnames(RandSVD) == (:compressor, :power_its, :rand_subspace)
        fieldtypes(RandSVD) == (Compressor, Int64, Bool)
        
        # test errors
        let compressor = TestCompressor(),
            power_its = -1,
            rand_subspace = false

            @test_throws ArgumentError(
                "Field `power_its` must be non-negative."
            ) RandSVD(compressor, power_its, rand_subspace)
        end

        # Test constructor
        let compressor = TestCompressor(),
            power_its = 2,
            rand_subspace = false,
            rf = RandSVD(compressor, power_its, rand_subspace) 

            @test typeof(rf.compressor) == TestCompressor
            @test rf.power_its == 2
            @test rf.rand_subspace == false
        end
    
    end

    @testset "RandSVD Recipe" begin
        @test_range_approximator RandSVDRecipe
        supertype(RandSVDRecipe) == ApproximatorRecipe

        # test the fieldnames and types
        @test fieldnames(RandSVDRecipe) == (
            :n_rows, :n_cols, :compressor, :power_its, :rand_subspace, :U, :S, :V
        )
        @test fieldtypes(RandSVDRecipe) == (
            Int64, 
            Int64, 
            CompressorRecipe, 
            Int64, 
            Bool, 
            AbstractArray, 
            AbstractVector, 
            AbstractArray
        )
    end
    
    @testset "RandSVD: Complete Approximator" begin
        # Test when correct cardinality, right, is specified
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RandSVD(compressor, 2, true)
            approx_rec = complete_approximator(approx, A)
            approx_rec.compressor.cardinality == Right()

            @test approx_rec.power_its == 2
            @test approx_rec.rand_subspace == true
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 5
        end
        
        # Test when wrong cardinality, left, is specified
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Left(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RandSVD(compressor, 2, false)
            @test_logs (:warn,
                "Compressor with cardinality `Left` being applied from `Right`."
            ) complete_approximator(approx, A)
            approx_rec = complete_approximator(approx, A)
            approx_rec.compressor.cardinality == Left()
            
            @test approx_rec.power_its == 2
            @test approx_rec.rand_subspace == false 
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 5
        end
        
    end

    @testset "RandSVD Recipe: rapproximate" begin
        # By testing the rapproximate function we also test rapproximate!
        # with power iterations
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RandSVD(compressor, 2, false)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
            # bound from theorem 9.1 of Halko Martinsson and Trop
            _,S,_ = svd(A)
            norm(S[1:5] - approx_rec.S) < norm(S[6:end])^2 + 
                norm(Diagonal(S[6:end]) * 
                approx_rec.compressor.op[6:end, :] * 
                pinv(approx_rec.compressor.op[1:5, :]))^2
        end

        # Test that compression_dim == n_rows gives a matrix that spans the range of A
        # with power iterations, this way the approximation is exact
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 10,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RandSVD(compressor, 2, false)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            approx_rec.compressor.cardinality == Right()
            @test approx_rec.power_its == 2
            @test approx_rec.rand_subspace == false
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 10 
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
            
            # Test that this spans the range and that the mul! function work
            @test norm(A - approx_rec * (approx_rec' * A)) < ATOL
        end

        # By testing the rapproximate function we also test rapproximate!
        # with subspace iterations
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 5,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RandSVD(compressor, 2, true)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
             # test bound from theorem 9.1 of Halko Martinsson and Trop
             _,S,_ = svd(A)
             norm(S[1:5] - approx_rec.S) < norm(S[6:end])^2 + 
                 norm(Diagonal(S[6:end]) * 
                 approx_rec.compressor.op[6:end, :] * 
                 pinv(approx_rec.compressor.op[1:5, :]))^2
        end

        # Test that compression_dim == n_rows gives a matrix that spans the range of A 
        # with subspace iterations, so that the approximation is exact
        let n_rows = 10,
            n_cols = 10,
            compression_dim = 10,
            cardinality = Right(),
            A = rand(n_rows, n_cols)

            compressor = TestCompressor(cardinality, compression_dim)
            approx = RandSVD(compressor, 2, true)
            approx_rec = rapproximate(approx, A)
            
            @test typeof(approx_rec.compressor) == TestCompressorRecipe  
            approx_rec.compressor.cardinality == Right()
            @test approx_rec.power_its == 2
            @test approx_rec.rand_subspace == true
            @test approx_rec.n_rows == 10
            @test approx_rec.n_cols == 10 
            # Check that the matrix is orthogonal
            gram_matrix = approx_rec.U' * approx_rec.U
            # check that the norm is 1, the diagonal is all 1
            @test opnorm(gram_matrix) ≈ 1
            # test that the diagonal is all ones
            diag_gram_matrix = diag(gram_matrix)
            for i = 1:compression_dim
                @test diag_gram_matrix[i] ≈ 1
            end
            
            # Test that this spans the range and that the mul! function work
            @test norm(A - approx_rec * (approx_rec' * A)) < ATOL
        end

    end

    @testset "RandSVD Recipe: mul!" begin
        n_rows = 10
        n_cols = 10
        compression_dim = 10 
        cardinality = Right()
        A = rand(n_rows, n_cols)
        C = rand(n_rows, n_cols)
        v = rand(n_cols)
        b = rand(n_cols)
        compressor = TestCompressor(cardinality, compression_dim)
        approx = RandSVD(compressor, 2, true)
        approx_rec = rapproximate(approx, A)
        # Check that the size function works
        size(approx_rec) == (approx_rec.n_rows, approx_rec.n_cols)
        size(approx_rec') == (approx_rec.n_cols, approx_rec.n_rows)
        # also test that the tranpose is the same as the parent
        ApproximatorAdjoint(approx_rec) == transpose(approx_rec)
        approx_rec = transpose(transpose(approx_rec))
        # test multiplication from the left
        let approx_rec = approx_rec, 
            A = A, 
            C = C,
            Cc = deepcopy(C) 

            mul!(C, approx_rec, A, 2.0, 1.0)
            @test C ≈ Cc + 2.0 * approx_rec.U * A
        end

        # test multiplication from the right
        let approx_rec = approx_rec, 
            A = A, 
            C = C,
            Cc = deepcopy(C) 

            mul!(C, A, approx_rec, 2.0, 1.0)
            @test C ≈ Cc + 2.0 * A * approx_rec.V'
        end

        # Test the vector multiplication
        # from the right
        let approx_rec = approx_rec, 
            v = v, 
            b = b,
            bc = deepcopy(b) 

            mul!(b, approx_rec, v, 2.0, 1.0)
            @test b ≈ bc + 2.0 * approx_rec.U * v
        end

        # Test the vector multiplication
        # from the left 
        let approx_rec = approx_rec, 
            v = v, 
            b = b,
            bc = deepcopy(b) 

            mul!(b', v', approx_rec, 2.0, 1.0)
            @test b ≈ (bc' + 2.0  * v' * approx_rec.V')'
        end

    end


end

end
