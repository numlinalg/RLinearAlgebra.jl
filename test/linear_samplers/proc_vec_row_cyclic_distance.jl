# This file is part of RLinearAlgebra

module ProceduralTestLSVRCyclicDistance

using Test, RLinearAlgebra

@testset "LSVR Cyclic Distance -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecRowDistCyclic) == LinSysVecRowSampler

    # Test whether the ordering is initialized correctly
    @test let
        cyc = LinSysVecRowDistCyclic()
        isempty(cyc.order)      
    end

    # Test whether the ordering is selected correctly 
    @test let
        A = ones(3,2)
        b = [1.0,2.0,3.0]
        x = zeros(2)

        cyc = LinSysVecRowDistCyclic()

        # Initialize Ordering and Pop First Vector
        flag = true         
        α, β = RLinearAlgebra.sample(cyc, A, b, x, 0)
        
        flag = flag & (β == view([3.0],1)) #Check first element
        flag = flag & (length(cyc.order) == 2) #Check length 
        flag = flag & (cyc.order == [2,1]) #Check ordering 

        flag
    end

    # Test whether ordering is appropriately initialized, popped 
    # and re-initialized
    @test let
        A = ones(3,2)
        b = [1.0,2.0,3.0]
        x = zeros(2)

        cyc = LinSysVecRowDistCyclic()

        # Initialize ordering and pop all vectors
        flag = true 
        for i = 0:2
            α, β = RLinearAlgebra.sample(cyc, A, b, x, i)
            flag = flag & (β == view(b,3-i))
        end

        # Check if order exhausted
        flag = flag & isempty(cyc.order)

        # Re-initialize (with new constant vector)
        b = [2.0, 3.0, 1.0]
        α, β = RLinearAlgebra.sample(cyc, A, b, x, 0)
        flag = flag & (β == view([3.0],1)) 
        flag = flag & (length(cyc.order) == 2)
        flag = flag & (cyc.order == [1,3])
        
        flag
    end

end

end