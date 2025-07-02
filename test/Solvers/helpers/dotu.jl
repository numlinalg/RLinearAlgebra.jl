module DotU
using Test, RLinearAlgebra, LinearAlgebra
import RLinearAlgebra: dotu
import LinearAlgebra: dot
@testset "Dotu" begin
    # begin by testing the size error
    let n = 10,
        a = ones(n),
        b = ones(n + 1)

        @test_throws DimensionMismatch(
            "Vector `a` and Vector `b` must be the same size."
        ) dotu(a, b)
    end
    
    # test that the dot product works for multiple types
    for type in [Int32, Int64, Float16, Float32, Float64, ComplexF32, ComplexF64]
        let n = 10,
            a = ones(type, n),
            b = ones(type, n)

            @test dot(conj(a), b) == dotu(a, b)
        end
    
    end

end

end
