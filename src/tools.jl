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
