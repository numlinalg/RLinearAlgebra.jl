module l2norm_distribution
using Test, RLinearAlgebra
using StatsBase: ProbabilityWeights

@testset "L2Norm" begin
    @testset "L2Norm: Distribution" begin
        # Verify supertypes, fieldnames and fieldtypes
        @test supertype(L2Norm) == Distribution
        @test fieldnames(L2Norm) == (:cardinality, :replace)
        @test fieldtypes(L2Norm) == (Cardinality, Bool)

        # Default
        let 
            u = L2Norm()
            @test u.cardinality == Undef()
            @test u.replace == false
        end

        let 
            u2 = L2Norm(cardinality = Left(), replace = true)
            @test u2.cardinality == Left()
            @test u2.replace == true
        end

        let 
            u3 = L2Norm(cardinality = Right(), replace = false)
            @test u3.cardinality == Right()
            @test u3.replace == false
        end

    end

    @testset "L2Norm: DistributionRecipe" begin
        # Verify supertypes, fieldnames and fieldtypes
        @test supertype(L2NormRecipe) == DistributionRecipe
        @test fieldnames(L2NormRecipe) == (:cardinality, :replace, :state_space, :weights)
        @test fieldtypes(L2NormRecipe) == (Cardinality, Bool, Vector{Int64}, ProbabilityWeights)
    end

    @testset "L2Norm: Complete Distribution" begin
        # Test with left compressor
        # Row 1: 1^2 + 0^2 = 1
        # Row 2: 0^2 + 2^2 = 4
        # Row 3: 1^2 + 1^2 = 2
        let A = [1.0 0.0; 0.0 2.0; 1.0 1.0], 
            u = L2Norm(cardinality = Left())
            
            ur = complete_distribution(u, A)
            @test ur.cardinality == Left()
            @test length(ur.state_space) == 3
            @test ur.weights ≈ ProbabilityWeights([1.0, 4.0, 2.0])
        end
        
        # Test with right compressor
        # Col 1: 1+0+1 = 2
        # Col 2: 0+4+1 = 5
        let A = [1.0 0.0; 0.0 2.0; 1.0 1.0], 
            u = L2Norm(cardinality = Right())

            ur = complete_distribution(u, A)
            @test ur.cardinality == Right()
            @test length(ur.state_space) == 2
            @test ur.weights ≈ ProbabilityWeights([2.0, 5.0])
        end

        # Test with undef compressor
        let A = randn(3, 5), 
            u = L2Norm(cardinality = Undef())

            @test_throws ArgumentError complete_distribution(u, A)
        end

    end

    @testset "L2Norm: Update Distribution" begin
        # Updating Right compression, changing dimensions
        let A = randn(3, 5),
            A2 = randn(4, 6),
            u = L2Norm(cardinality = Right()),
            ur = complete_distribution(u, A)
            
            update_distribution!(ur, A2)
            @test ur.cardinality == Right()
            @test ur.state_space == collect(1:6)
            @test length(ur.weights) == 6
            @test ur.weights ≈ ProbabilityWeights(vec(sum(abs2, A2, dims=1)))
        end

        # Updating Left compression, changing dimensions
        let A = randn(5, 3),
            A2 = randn(6, 4),
            u = L2Norm(cardinality = Left()),
            ur = complete_distribution(u, A)
            
            update_distribution!(ur, A2)
            @test ur.cardinality == Left()
            @test ur.state_space == collect(1:6)
            @test length(ur.weights) == 6
            @test ur.weights ≈ ProbabilityWeights(vec(sum(abs2, A2, dims=2)))
        end

        # Updating without changing dimensions
        let A = [1.0 0.0; 0.0 1.0],     # Weights [1, 1]
            A2 = [2.0 1.0; 1.0 2.0],    # Weights [5, 5]
            u = L2Norm(cardinality = Left()),
            ur = complete_distribution(u, A)

            # Pre-check
            @test ur.weights ≈ ProbabilityWeights([1.0, 1.0])

            update_distribution!(ur, A2)
            @test ur.weights ≈ ProbabilityWeights([5.0, 5.0])
            # Check state space is still valid size
            @test length(ur.state_space) == 2
        end

        # Error handling
        let A = randn(5, 3),
            A2 = randn(6, 4),
            card = Undef(),
            replace = true,
            state_space = collect(1:5),
            weights = ProbabilityWeights(ones(5)),
            ur = L2NormRecipe(card, replace, state_space, weights)
            
            @test_throws ArgumentError update_distribution!(ur, A2)
        end       

    end

    @testset "L2Norm: Sample Distribution" begin
        # All zero row
        # Row 1: [1, 0] -> norm^2 = 1
        # Row 2: [0, 0] -> norm^2 = 0  <-- Should never be sampled
        # Row 3: [1, 1] -> norm^2 = 2
        let A = [1.0 0.0; 0.0 0.0; 1.0 1.0],
            x = zeros(Int, 100), # Sample 100 times
            replace = true,
            u = L2Norm(cardinality = Left(), replace = replace),
            ur = complete_distribution(u, A)

            # Check weights are correct
            @test ur.weights ≈ ProbabilityWeights([1.0, 0.0, 2.0])
            sample_distribution!(x, ur)

            # Verify that index 2 (the zero row) is NEVER sampled
            @test 2 ∉ x
            # Verify that valid indices (1 and 3) ARE sampled (statistically likely in 100 tries)
            @test any(==(1), x)
            @test any(==(3), x)
        end

        # All zero column
        # Col 1: [1, 1] -> norm^2 = 2
        # Col 2: [0, 0] -> norm^2 = 0 <-- Should never be sampled
        let A = [1.0 0.0; 1.0 0.0],
            x = zeros(Int, 100),
            replace = true,
            u = L2Norm(cardinality = Right(), replace = replace),
            ur = complete_distribution(u, A)

            # Check weights
            @test ur.weights ≈ ProbabilityWeights([2.0, 0.0])

            # Perform sampling
            sample_distribution!(x, ur)

            # Verify that index 2 (the zero col) is NEVER sampled
            @test 2 ∉ x
            # Verify that index 1 IS sampled
            @test all(==(1), x)
        end

        # Test if the sample_distribution! returns valid indices
        let A = randn(3, 20),
            x = zeros(Int, 10),
            u = L2Norm(cardinality = Right()),
            ur = complete_distribution(u, A)
            
            sample_distribution!(x, ur)
            @test ur.cardinality == Right()
            @test all(s -> 1 <= s <= 20, x)
        end

        let A = randn(20, 3),
            x = zeros(Int, 10),
            u = L2Norm(cardinality = Left()),
            ur = complete_distribution(u, A)
            
            sample_distribution!(x, ur)
            @test ur.cardinality == Left()
            @test all(s -> 1 <= s <= 20, x)
        end

    end

end

end