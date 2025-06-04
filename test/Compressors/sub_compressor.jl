module sub_compressor
using Test, RLinearAlgebra, Random
using StatsBase: ProbabilityWeights, sample
import LinearAlgebra: mul!, Adjoint
using ..FieldTest
using ..ApproxTol

Random.seed!(2131)
@testset "Sub_Compressor" begin
    @testset "Sub_Compressor: Compressor" begin
        # Verify Supertype
        @test supertype(SubCompressor) == Compressor

        # Verify fields and types
        @test fieldnames(SubCompressor) == (:cardinality, :compression_dim, :distribution)
        @test fieldtypes(SubCompressor) == (Cardinality, Int64, Distribution)

        # Verify the Internal Constructor
        let cardinality = Left(), compression_dim = 0, distribution = Uniform()
            @test_throws ArgumentError(
                "Field `compression_dim` must be positive."
            ) SubCompressor(
                cardinality, compression_dim, distribution
            )
        end

        # Default values
        let sc_default = SubCompressor()
            @test sc_default.cardinality == Left()
            @test sc_default.compression_dim == 2
            @test sc_default.distribution isa Uniform
            # Check defaults of the default Uniform distribution instance
            @test sc_default.distribution.cardinality == Undef()
            @test sc_default.distribution.replace == false
        end

        # Specific cardinality
        let card_val = Right()
            dist_instance = Uniform(cardinality=card_val) 
            compressor = SubCompressor(; cardinality=card_val, distribution=dist_instance)
            @test compressor.cardinality == card_val
            @test compressor.distribution.cardinality == card_val
        end

        let card_val = Left()
            dist_instance = Uniform(cardinality=card_val)
            compressor = SubCompressor(; cardinality=card_val, distribution=dist_instance)
            typeof(compressor.cardinality) == Cardinality
            @test compressor.cardinality == card_val
            @test compressor.distribution.cardinality == card_val
        end

        # Test with specific compression_dim
        let comp_dim_val = 15
            compressor = SubCompressor(; compression_dim=comp_dim_val)
            @test compressor.compression_dim == comp_dim_val
        end

        # Test with specific distribution instance
        let my_dist = Uniform(cardinality=Right(), replace=true)
            compressor = SubCompressor(; distribution=my_dist)
            @test compressor.distribution === my_dist
            @test compressor.distribution.cardinality == Right()
            @test compressor.distribution.replace == true
            # Default cardinality and compression_dim should be used
            @test compressor.cardinality == Left()
            @test compressor.compression_dim == 2
        end

        # Test combination of specific arguments
        let final_card = Right(), final_cdim = 22
            final_dist = Uniform(cardinality=final_card, replace=false)
            compressor = SubCompressor(; cardinality=final_card, compression_dim=final_cdim, distribution=final_dist)
            @test compressor.cardinality == final_card
            @test compressor.compression_dim == final_cdim
            @test compressor.distribution === final_dist
        end
    end

    @testset "Sub_Compressor: CompressorRecipe" begin
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

    end

    # @testset "SubCompressor: get_dims" begin
    #     let comp_dim_left = 4, 
    #         card_left = Left(),
    #         A_matrix = randn(10, 7),
    #         A_rows = size(A_matrix, 1),
    #         A_cols = size(A_matrix, 2)

    #         n_r, n_c, init_s = get_dims(comp_dim_left, card_left, A_matrix)

    #         @test n_r == comp_dim_left
    #         @test n_c == A_rows
    #         @test init_s == A_rows
    #     end

    #     let comp_dim_right = 3, 
    #         card_right = Right(),
    #         A_matrix = randn(10, 7),
    #         A_rows = size(A_matrix, 1),
    #         A_cols = size(A_matrix, 2)

    #         n_r, n_c, init_s = get_dims(comp_dim_right, card_right, A_matrix)

    #         @test n_r == A_cols
    #         @test n_c == comp_dim_right
    #         @test init_s == A_cols
    #     end
    # end

    @testset "SubCompressor: complete_compressor" begin
        let card_instance = Left(),
            a_matrix_rows = 10, 
            a_matrix_cols = 8,  
            comp_dim = 4,  
            replace_sampling = false 

            A = randn(a_matrix_rows, a_matrix_cols)

            # Create the distribution instance. Its cardinality will be updated by complete_compressor
            dist_instance = Uniform(cardinality=Undef(), replace=replace_sampling)
            
            sub_comp_settings = SubCompressor(cardinality=card_instance, compression_dim=comp_dim, distribution=dist_instance)
            
            # First, check it's not already set to card_instance
            if dist_instance.cardinality != card_instance
                @test dist_instance.cardinality != card_instance 
            end

            recipe = complete_compressor(sub_comp_settings, A)

            # subcompressor.distribution.cardinality should be updated
            @test dist_instance.cardinality == card_instance
            
            # Test the recipe values and types. 
            # For Left, recipe.n_rows = compression dim, recipe.n_cols = A's rows
            @test recipe.cardinality == card_instance
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == comp_dim       
            @test recipe.n_cols == a_matrix_rows  
            
            @test recipe.distribution_recipe isa UniformRecipe
            dist_recipe_concrete = recipe.distribution_recipe::UniformRecipe
            @test dist_recipe_concrete.cardinality == card_instance
            @test dist_recipe_concrete.replace == replace_sampling
            @test dist_recipe_concrete.state_space == collect(1:a_matrix_rows)
            @test dist_recipe_concrete.weights.values ≈ ones(a_matrix_rows) 

            @test length(recipe.idx) == comp_dim
            @test eltype(recipe.idx) == Int64

            # Check indices inside rows of A
            @test all(x -> 1 <= x <= a_matrix_rows, recipe.idx) 

            # Test no replacement
            @test length(unique(recipe.idx)) == comp_dim
            @test recipe.idx_v == view(recipe.idx, :)
            @test parent(recipe.idx_v) === recipe.idx
        end

        let card_instance = Right(),
            a_matrix_rows = 12, 
            a_matrix_cols = 15, 
            comp_dim = 5,         
            replace_sampling = true

            # Define A inside the let block
            A = randn(a_matrix_rows, a_matrix_cols)
            
            # Create the distribution instance
            dist_instance = Uniform(cardinality=Undef(), replace=replace_sampling)
            
            sub_comp_settings = SubCompressor(cardinality=card_instance, compression_dim=comp_dim, distribution=dist_instance)

            # The distribution's cardinality should not be the same
            if dist_instance.cardinality != card_instance
                @test dist_instance.cardinality != card_instance
            end
            
            recipe = complete_compressor(sub_comp_settings, A)

            # Verify the update
            @test dist_instance.cardinality == card_instance

            # Test the recipe values and types
            # For Right, recipe.n_rows =  A's columns, recipe.n_cols = compression dim
            @test recipe.cardinality == card_instance
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == a_matrix_cols  
            @test recipe.n_cols == comp_dim        
            
            @test recipe.distribution_recipe isa UniformRecipe
            dist_recipe_concrete = recipe.distribution_recipe::UniformRecipe
            @test dist_recipe_concrete.cardinality == card_instance
            @test dist_recipe_concrete.replace == replace_sampling
            @test dist_recipe_concrete.state_space == collect(1:a_matrix_cols)
            @test dist_recipe_concrete.weights.values ≈ ones(a_matrix_cols) 

            @test length(recipe.idx) == comp_dim
            @test eltype(recipe.idx) == Int64

            # Check indices inside cols of A
            @test all(x -> 1 <= x <= a_matrix_cols, recipe.idx)
            @test recipe.idx_v == view(recipe.idx, :)
            @test parent(recipe.idx_v) === recipe.idx
        end
    end

    @testset "SubCompressor: update_compressor!" begin
        let card_instance = Left(),
            a_matrix_rows = 100, 
            a_matrix_cols = 15,  
            comp_dim = 10,          
            replace_sampling = false 

            A = randn(a_matrix_rows, a_matrix_cols)
            x_dummy = randn(a_matrix_cols) 
            b_dummy = randn(a_matrix_rows) 
            
            dist_instance = Uniform(cardinality=card_instance, replace=replace_sampling)
            sub_comp_settings = SubCompressor(cardinality=card_instance, compression_dim=comp_dim, distribution=dist_instance)
            recipe = complete_compressor(sub_comp_settings, A)
            
            # Store the old indices to verify they change
            old_idx = deepcopy(recipe.idx)
            # Store old distribution recipe state if it's supposed to change and be testable
            # For Uniform, update_distribution! re-initializes based on A

            update_compressor!(recipe, A, x_dummy, b_dummy)

            # After updates, index should be changed
            @test recipe.idx != old_idx
            
            # Test invariant properties of the recipe after update
            @test recipe.cardinality == card_instance
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == comp_dim
            @test recipe.n_cols == a_matrix_rows
            
            @test recipe.distribution_recipe isa UniformRecipe 
            dist_recipe_concrete = recipe.distribution_recipe::UniformRecipe
            @test dist_recipe_concrete.cardinality == card_instance 
            @test dist_recipe_concrete.replace == replace_sampling
            @test dist_recipe_concrete.state_space == collect(1:a_matrix_rows) 
            @test dist_recipe_concrete.weights.values ≈ ones(a_matrix_rows)

            @test length(recipe.idx) == comp_dim
            @test eltype(recipe.idx) == Int64
            @test all(val -> 1 <= val <= a_matrix_rows, recipe.idx)
            if !replace_sampling
                @test length(unique(recipe.idx)) == comp_dim
            end
            @test recipe.idx_v == view(recipe.idx, :)
            @test parent(recipe.idx_v) === recipe.idx
        end

        let card_instance = Right(),
            a_matrix_rows = 15,
            a_matrix_cols = 100,
            comp_dim = 10,
            replace_sampling = false

            A = randn(a_matrix_rows, a_matrix_cols)
            x_dummy = randn(a_matrix_cols)
            b_dummy = randn(a_matrix_rows)

            dist_instance = Uniform(cardinality=card_instance, replace=replace_sampling)
            sub_comp_settings = SubCompressor(cardinality=card_instance, compression_dim=comp_dim, distribution=dist_instance)
            recipe = complete_compressor(sub_comp_settings, A)

            old_idx = deepcopy(recipe.idx)

            update_compressor!(recipe, A, x_dummy, b_dummy)

            # After updates, index should be changed
            @test recipe.idx != old_idx
    
            # Test unchanged values
            @test recipe.cardinality == card_instance
            @test recipe.compression_dim == comp_dim
            @test recipe.n_rows == a_matrix_cols
            @test recipe.n_cols == comp_dim

            @test recipe.distribution_recipe isa UniformRecipe
            dist_recipe_concrete = recipe.distribution_recipe::UniformRecipe
            @test dist_recipe_concrete.cardinality == card_instance
            @test dist_recipe_concrete.replace == replace_sampling
            @test dist_recipe_concrete.state_space == collect(1:a_matrix_cols)
            @test dist_recipe_concrete.weights.values ≈ ones(a_matrix_cols) 

            @test length(recipe.idx) == comp_dim
            @test eltype(recipe.idx) == Int64
            @test all(val -> 1 <= val <= a_matrix_cols, recipe.idx)
            if !replace_sampling
                @test length(unique(recipe.idx)) == comp_dim
            end
            @test recipe.idx_v == view(recipe.idx, :)
            @test parent(recipe.idx_v) === recipe.idx
        end
    end


    @testset "SubCompressor: Multiplication (mul!)" begin
        @testset "Left Multiplication: C = beta*C + alpha*(S*A)" begin
            let a_matrix_rows = 20, 
                a_matrix_cols = 12, 
                comp_dim = 7,      
                alpha_val = 2.5, 
                beta_val = 1.5

                A_val = randn(a_matrix_rows, a_matrix_cols)

                sc_left = SubCompressor(
                    cardinality=Left(),
                    compression_dim=comp_dim,
                    distribution=Uniform(cardinality=Left(), replace=false)
                )
                S_recipe_left = complete_compressor(sc_left, A_val)

                # Output matrix C
                C_output = randn(comp_dim, a_matrix_cols)
                C_original = deepcopy(C_output) 

                # Ground truth for S*A (row sub-sampling of A)
                SA_exact = A_val[S_recipe_left.idx, :]

                # Perform the multiplication using the function under test
                mul!(C_output, S_recipe_left, A_val, alpha_val, beta_val)

                # Verify the result
                expected_C = alpha_val * SA_exact + beta_val * C_original
                @test C_output ≈ expected_C
            end
        end

        @testset "Right Multiplication: C = beta*C + alpha*(A*S)" begin
            let a_matrix_rows = 18,
                a_matrix_cols = 22,
                comp_dim = 6,   
                alpha_val = 2.5, 
                beta_val = 1.5

                A_val = randn(a_matrix_rows, a_matrix_cols)

                # Setup SubCompressor and its recipe for Right compression
                sc_right = SubCompressor(
                    cardinality=Right(),
                    compression_dim=comp_dim,
                    distribution=Uniform(cardinality=Right(), replace=false)
                )
                S_recipe_right = complete_compressor(sc_right, A_val)

                # Output matrix C
                C_output = randn(a_matrix_rows, comp_dim)
                C_original = deepcopy(C_output) # For checking the beta part

                # Ground truth for A*S (column sub-sampling of A)
                AS_exact = A_val[:, S_recipe_right.idx]

                # Perform the multiplication using the function under test
                mul!(C_output, A_val, S_recipe_right, alpha_val, beta_val)

                # Verify the result
                expected_C = alpha_val * AS_exact + beta_val * C_original
                @test C_output ≈ expected_C
            end
        end
    end
end

end