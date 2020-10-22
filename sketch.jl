using LinearAlgebra

function tail_energy(A, j)
    F = svd(A)
    return sum(F.S[j:end])
end

m = 1000
n = 5

A = rand(-5.0:0.01:5.0, m, n)

# define sketch parameters
r = 3 # target range
k = r # range sketch
l = k # co-range sketch

# Test matrices
Ω = randn(n, k)
Ψ = randn(l, m)

# Create sketch
Y = A*Ω
W = Ψ*A

# Low-Rank approximation
Q, _ = qr(Y)
X = (Ψ*Q)\W
B = Q*X

# check validity
AA = A'*A
BB = B'*B

# diagnosis
println(norm(A - B))

