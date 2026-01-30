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
        let u = L2Norm()
            @test u.cardinality == Undef()
            @test u.replace == true
        end

        let u2 = L2Norm(cardinality = Left(), replace = false)
            @test u2.cardinality == Left()
            @test u2.replace == false
        end

        let u3 = L2Norm(cardinality = Right(), replace = true)
            @test u3.cardinality == Right()
            @test u3.replace == true
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
            u = L2Norm(cardinality = Left()),
            ur = complete_distribution(u, A)

            @test ur.cardinality == Left()
            @test length(ur.state_space) == 3
            @test ur.weights ≈ ProbabilityWeights([1.0, 4.0, 2.0])
        end
        
        # Test with right compressor
        # Col 1: 1+0+1 = 2
        # Col 2: 0+4+1 = 5
        let A = [1.0 0.0; 0.0 2.0; 1.0 1.0], 
            u = L2Norm(cardinality = Right()),
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

        # Updating without changing dimensions (Testing optimization path)
        let A = [1.0 0.0; 0.0 1.0],     # Weights [1, 1]
            A2 = [2.0 1.0; 1.0 2.0],    # Weights [5, 5]
            u = L2Norm(cardinality = Left()),
            ur = complete_distribution(u, A)

            # Pre-check
            @test ur.weights ≈ ProbabilityWeights([1.0, 1.0])
            
            # Update
            update_distribution!(ur, A2)
            
            # Check weights updated
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
        # Test if the sample_distribution! returns valid indices
        let A = randn(3, 5),
            x = zeros(Int, 10),
            u = L2Norm(cardinality = Right()),
            ur = complete_distribution(u, A)
            
            sample_distribution!(x, ur)
            @test ur.cardinality == Right()
            @test all(s -> 1 <= s <= 5, x)
        end

    end

end

end