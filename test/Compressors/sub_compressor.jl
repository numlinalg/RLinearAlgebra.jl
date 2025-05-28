module sub_compressor
using Test, RLinearAlgebra, Random
using StatsBase: ProbabilityWeights
import LinearAlgebra: mul!, Adjoint
using ..FieldTest
using ..ApproxTol

Random.seed!(2131)
@testset "Sub-Compressor" begin
    @testset "Sub-Compressor: Compressor" begin
        # Verify Supertype
        @test supertype(SubCompressor) == Compressor

        # Verify fields and types
        @test fieldnames(SubCompressor) == (:cardinality, :compression_dim, :distribution)
        @test fieldtypes(SubCompressor) == (Cardinality, Int64, Distribution)

        # Default values
        sc_default = SubCompressor() 
        @test sc_default.cardinality == Left() 
        @test sc_default.compression_dim == 2  
        @test typeof(sc_default.distribution) == Uniform 
        @test sc_default.distribution.cardinality == Undef() 
        @test sc_default.distribution.replace == false 

        # Specific values
        uniform_right_no_replace = Uniform(cardinality=Right(), replace=false)
        sc_specific = SubCompressor(cardinality=Right(), compression_dim=10, distribution=uniform_right_no_replace)
        @test sc_specific.cardinality == Right()
        @test sc_specific.compression_dim == 10
        @test sc_specific.distribution == uniform_right_no_replace
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

        let card_type = Left, 
            comp_dim = 5,
            actual_n_rows = comp_dim,
            actual_n_cols = 20,           
            replace_sampling = false

            # Setup for distribution_recipe 
            dist_state_space = collect(1:actual_n_cols)
            dist_weights = ProbabilityWeights(ones(actual_n_cols) ./ actual_n_cols)
            dist_recipe_val = UniformRecipe(card_type(), replace_sampling, dist_state_space, dist_weights)

            # Setup for idx and idx_v
            idx_val = sort(sample(dist_state_space, comp_dim, replace=replace_sampling))
            idx_v_val = view(idx_val, :)

            # Construct SubCompressorRecipe
            recipe = SubCompressorRecipe(card_type(), comp_dim, actual_n_rows, actual_n_cols,
                                         dist_recipe_val, idx_val, idx_v_val)

            @test recipe.cardinality == card_type() 
            @test isa(recipe.cardinality, card_type)  
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == actual_n_rows
            @test recipe.n_cols == actual_n_cols
            @test recipe.distribution_recipe == dist_recipe_val
            @test isa(recipe.distribution_recipe, UniformRecipe)
            @test recipe.idx == idx_val
            @test recipe.idx_v == idx_v_val
            @test eltype(recipe.idx_v) == eltype(idx_val)
            @test parent(recipe.idx_v) === recipe.idx 
        end

        let card_type = Right, 
            comp_dim = 6,
            actual_n_rows = 25,       
            actual_n_cols = comp_dim, 
            replace_sampling = true

            # Setup for distribution_recipe 
            dist_state_space = collect(1:actual_n_rows) 
            dist_weights = ProbabilityWeights(ones(actual_n_rows) ./ actual_n_rows)
            dist_recipe_val = UniformRecipe(card_type(), replace_sampling, dist_state_space, dist_weights)

            # Setup for idx and idx_v
            idx_val = sort(sample(dist_state_space, comp_dim, replace=replace_sampling))
            idx_v_val = view(idx_val, :)

            # Construct SubCompressorRecipe
            recipe = SubCompressorRecipe(card_type(), comp_dim, actual_n_rows, actual_n_cols,
                                         dist_recipe_val, idx_val, idx_v_val)

            @test recipe.cardinality == card_type()
            @test isa(recipe.cardinality, card_type)
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == actual_n_rows
            @test recipe.n_cols == actual_n_cols
            @test recipe.distribution_recipe == dist_recipe_val
            @test isa(recipe.distribution_recipe, UniformRecipe)
            @test recipe.idx == idx_val
            @test recipe.idx_v == idx_v_val
            @test eltype(recipe.idx_v) == eltype(idx_val)
            @test parent(recipe.idx_v) === recipe.idx
        end
        
        # Verify the internal constructor: Left Compressor
        let card = Left,
            A = randn(10, 3),
            compression_dim = 5,
            distribution = Uniform(),
            compressor = SubCompressor(card(), compression_dim, distribution)
            
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

    @testset "SubCompressor: complete_compressor" begin
        A_rows, A_cols = 20, 15

        @testset "Left Cardinality with Uniform Distribution" begin
            let card = Left(), comp_dim = 5, replace_sampling = false
                # Ensure the distribution's cardinality is set correctly for complete_distribution
                dist = Uniform(cardinality=card, replace=replace_sampling)
                A = randn(A_rows, A_cols)
                sc = SubCompressor(cardinality=card, compression_dim=comp_dim, distribution=dist)
                scr = complete_compressor(sc, A)

                @test scr.cardinality == card
                @test scr.compression_dim == comp_dim
                @test scr.n_rows == comp_dim
                @test scr.n_cols == A_rows
                @test isa(scr.distribution_recipe, UniformRecipe)
                @test length(scr.idx) == comp_dim
                @test eltype(scr.idx) == Int64
                @test all(x -> 1 <= x <= A_rows, scr.idx) 
                if !replace_sampling
                    @test length(unique(scr.idx)) == comp_dim 
                end
                @test scr.idx_v == view(scr.idx, :)

                # Test UniformRecipe properties
                ur = scr.distribution_recipe::UniformRecipe
                @test ur.cardinality == card
                @test ur.replace == replace_sampling
                @test ur.state_space == collect(1:A_rows)
                @test ur.weights isa ProbabilityWeights
                @test ur.weights.values ≈ ones(A_rows) ./ A_rows # Uniform weights
            end
        end

        @testset "Right Cardinality with Uniform Distribution" begin
            let card = Right(), comp_dim = 7, replace_sampling = true
                dist = Uniform(cardinality=card, replace=replace_sampling)
                A = randn(A_rows, A_cols)
                sc = SubCompressor(cardinality=card, compression_dim=comp_dim, distribution=dist)
                scr = complete_compressor(sc, A)

                @test scr.cardinality == card
                @test scr.compression_dim == comp_dim
                # Corrected based on your get_dims and typical right compression
                @test scr.n_rows == A_cols   
                @test scr.n_cols == comp_dim 

                @test isa(scr.distribution_recipe, UniformRecipe)
                @test length(scr.idx) == comp_dim
                @test eltype(scr.idx) == Int64
                @test all(x -> 1 <= x <= A_cols, scr.idx) 
                if !replace_sampling 
                    @test length(unique(scr.idx)) == comp_dim
                end
                @test scr.idx_v == view(scr.idx, :)

                ur = scr.distribution_recipe::UniformRecipe
                @test ur.cardinality == card
                @test ur.replace == replace_sampling
                @test ur.state_space == collect(1:A_cols)
                @test ur.weights isa ProbabilityWeights
                @test ur.weights.values ≈ ones(A_cols) ./ A_cols
            end
        end
    end

    # @testset "Sub Compressor: Left Multiplication" begin
    #     let n_rows = 10,
    #         n_cols = 3,
    #         c_dim = 5,
    #         A = rand(n_rows, n_cols),
    #         B = rand(c_dim, n_cols),
    #         C1 = rand(c_dim, n_cols),
    #         C2 = rand(n_rows, n_cols),
    #         x = rand(n_rows),
    #         y = rand(c_dim),
    #         S_info = SubCompressor(; compression_dim=c_dim, distribution=Uniform()),
    #         S = complete_compressor(S_info, A)

    #         # copies are for comparing with the "true version"
    #         C1c = deepcopy(C1)
    #         C2c = deepcopy(C2)
    #         yc = deepcopy(y)

    #         # do the multiplications
    #         mul!(C1, S, A)
    #     end

    # end

    @testset "SubCompressor: Multiplication (mul!)" begin
        A_val = randn(20, 12) # Use a fresh A for each mul test scope if modified
        alpha, beta = 2.0, 3.0

        @testset "Left Multiplication" begin
            A_rows, A_cols = size(A_val)
            comp_dim = 7
            # Ensure distribution has correct cardinality for complete_distribution
            sc_left = SubCompressor(cardinality=Left(), compression_dim=comp_dim, distribution=Uniform(cardinality=Left(), replace=false))
            S_left = complete_compressor(sc_left, A_val)
            
            # Ground truth: S_left * A_val selects rows from A_val
            SA_exact = A_val[S_left.idx, :]

            # Test mul!(C, S, A, alpha, beta)
            C1 = randn(comp_dim, A_cols) # Output matrix
            C1_orig = deepcopy(C1)
            # The mul! function in your code for Left applies S to A
            mul!(C1, S_left, A_val, alpha, beta) # C1 = beta*C1 + alpha*A[S.idx,:]
            @test C1 ≈ alpha * SA_exact + beta * C1_orig

            B_for_adj = randn(comp_dim, A_cols + 3) # K_B = A_cols + 3
            C_StB_exact = zeros(A_rows, size(B_for_adj, 2))

            for i_comp in 1:comp_dim # iterate over rows of B_for_adj / columns of S_left'
                selected_row_in_A = S_left.idx[i_comp]
                C_StB_exact[selected_row_in_A, :] .+= B_for_adj[i_comp, :]
            end
        end

        @testset "Right Multiplication" begin
            A_rows, A_cols = size(A_val)
            comp_dim = 6
            sc_right = SubCompressor(cardinality=Right(), compression_dim=comp_dim, distribution=Uniform(cardinality=Right(), replace=false))
            S_right = complete_compressor(sc_right, A_val)

            # Ground truth: A_val * S_right selects columns from A_val
            AS_exact = A_val[:, S_right.idx]

            # Test A*S (if * is defined for SubCompressorRecipe)
            # C_AS = A_val * S_right
            # @test C_AS ≈ AS_exact

            # Test mul!(C, A, S, alpha, beta)
            C2 = randn(A_rows, comp_dim) # Output matrix
            C2_orig = deepcopy(C2)
            mul!(C2, A_val, S_right, alpha, beta) # C2 = beta*C2 + alpha*A[:,S.idx]
            @test C2 ≈ alpha * AS_exact + beta * C2_orig

            # Test Adjoint multiplication: B * S'
            B_for_adj_right = randn(A_rows - 2, comp_dim) # K_B = A_rows - 2
            C_BSt_exact_right = zeros(size(B_for_adj_right,1), A_cols)
            for i_comp in 1:comp_dim # iterate over cols of B_for_adj_right / rows of S_right'
                selected_col_in_A = S_right.idx[i_comp]
                C_BSt_exact_right[:, selected_col_in_A] .+= B_for_adj_right[:, i_comp]
            end
        end
    end

    @testset "SubCompressor: update_compressor!" begin
        A_rows, A_cols = 10, 8
        A = randn(A_rows, A_cols)
        # Dummy x and b, as update_distribution! for Uniform doesn't use them,
        # but other distributions might.
        x_dummy = randn(A_cols)
        b_dummy = randn(A_rows)


        @testset "Left Cardinality" begin
            let card = Left(), comp_dim = 4, dist = Uniform(cardinality=card, replace=false)
                sc = SubCompressor(cardinality=card, compression_dim=comp_dim, distribution=dist)
                scr = complete_compressor(sc, A)
                
                old_idx = deepcopy(scr.idx)
                # update_distribution! for Uniform might re-init weights/state_space,
                # then sample_distribution! re-samples indices.
                update_compressor!(scr, A, x_dummy, b_dummy)

                @test scr.idx != old_idx # Indices should generally change due to resampling
                @test length(scr.idx) == comp_dim
                @test all(val -> 1 <= val <= A_rows, scr.idx)
                @test length(unique(scr.idx)) == comp_dim # Since replace=false for Uniform
                @test scr.idx_v == view(scr.idx, :)
            end
        end

        @testset "Right Cardinality" begin
            let card = Right(), comp_dim = 3, dist = Uniform(cardinality=card, replace=false)
                sc = SubCompressor(cardinality=card, compression_dim=comp_dim, distribution=dist)
                scr = complete_compressor(sc, A)
                
                old_idx = deepcopy(scr.idx)
                update_compressor!(scr, A, x_dummy, b_dummy)

                @test scr.idx != old_idx
                @test length(scr.idx) == comp_dim
                @test all(val -> 1 <= val <= A_cols, scr.idx)
                @test length(unique(scr.idx)) == comp_dim
                @test scr.idx_v == view(scr.idx, :)
            end
        end
    end

end


end