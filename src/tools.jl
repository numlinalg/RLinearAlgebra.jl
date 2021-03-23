using Random

#Generate a matrix of given rank and ratio kappa between
#largest and smallest nonzero singular values
function generate_matrix(
    m :: Int,       #Number of rows
    n :: Int,       #Number of columns
    r :: Int,       #Rank
    κ :: Float64    #Condition number
)
    #Compute Rotations
    u, _ = qr(randn(m, m))
    vt, _ = qr(randn(n, n))

    #Compute singular values
    sig = sort(rand(r), rev=true) |> λ -> λ./maximum(λ)
    a = (1 - 1/κ)/(1 - minimum(sig))
    b = 1 - a
    σ =  a.*sig .+ b
    (min(m, n) > r) && append!(σ, zeros(min(m, n) - r))

    #Return Matrix
    return u*diagm(m, n, 0 => σ)*vt
end

# Generate a set of isotropic vector
function isotropic_vector(n::Int)
    return randn(n)
end

function isotropic_vector!(x::AbstractVector)
    randn!(x)
    return nothing
end

function randomized_trace(A::AbstractMatrix, nsamples::Int)
    n = size(A, 1)
    w = zeros(n)
    x = zeros(n)
    estimator = 0.0
    # TODO: use compensated summation technique for large nsamples
    factor = 1.0/nsamples

    for i=1:nsamples
        isotropic_vector!(w)
        x .= A*w
        estimator += factor*(w'*x)
    end

    return estimator
end
