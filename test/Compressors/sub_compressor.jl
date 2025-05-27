module sub_compressor
using Test, RLinearAlgebra, Random
using StatsBase: ProbabilityWeights
import LinearAlgebra: mul!, Adjoint
using ..FieldTest
using ..ApproxTol

@testset "Sub-Compressor" begin
    @testset "Sub-Compressor: Compressor" begin
        # Verify Supertype
        @test supertype(SubCompressor) == Compressor

        # Verify fields and types
        @test fieldnames(SubCompressor) == (:cardinality, :compression_dim, :distribution)
        @test fieldtypes(SubCompressor) == (Cardinality, Int64, Distribution)
    end

    @testset "Sub-Compressor: CompressorRecipe" begin
        @test_compressor SubCompressorRecipe
        @test fieldnames(SubCompressorRecipe) == (
            :cardinality,
            :compression_dim,
            :n_rows,
            :n_cols,
            :distribution_recipe,
            :idx,
            :idx_v
        )
        @test fieldtypes(SubCompressorRecipe) == (
            Cardinality,
            Int64,
            Int64,
            Int64,
            DistributionRecipe,
            Vector{Int64},
            SubArray
        )

        # Verify the internal constructor: Left Compressor
        let card = Left,
            A = randn(10, 3),
            compression_dim = 5,
            distribution = Uniform(),
            compressor = SubCompressor(card(), compression_dim, distribution),
            compressor_recipe = complete_compressor(compressor, A)
            @test typeof(compressor_recipe.cardinality) == card
            @test compressor_recipe.compression_dim == compression_dim
            @test compressor_recipe.n_rows == compression_dim
            @test compressor_recipe.n_cols == size(A, 1)
            @test compressor_recipe.distribution_recipe.cardinality == card() ############### Change it!
            @test compressor_recipe.distribution_recipe.state_space == collect(1:10)
            @test compressor_recipe.distribution_recipe.weights == ProbabilityWeights(ones(10))
            @test length(compressor_recipe.idx) == compression_dim
            @test eltype(compressor_recipe.idx) == Int64
        end

    end

    @testset "Sub Compressor: Left Multiplication" begin
        let n_rows = 10,
            n_cols = 3,
            c_dim = 5,
            A = rand(n_rows, n_cols),
            B = rand(c_dim, n_cols),
            C1 = rand(c_dim, n_cols),
            C2 = rand(n_rows, n_cols),
            x = rand(n_rows),
            y = rand(c_dim),
            S_info = SubCompressor(; compression_dim=c_dim, distribution=Uniform()),
            S = complete_compressor(S_info, A)

            # copies are for comparing with the "true version"
            C1c = deepcopy(C1)
            C2c = deepcopy(C2)
            yc = deepcopy(y)

            # do the multiplications
            mul!(C1, S, A)
        end

    end

end


end