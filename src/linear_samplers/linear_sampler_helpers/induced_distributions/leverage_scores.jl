# Date: 12/19/2024
# Author: Christian Varner
# Purpose: Create functionality for distributions for rows and columns based
# on the leverage scores.

"""
    TODO - documentation
"""
function leverage_score_distribution(
    A::AbstractMatrix,
    row_distribution::Bool
)
    # error checking
    @assert row_distribution ? (size(A)[1] >= size(A)[2]) : (size(A')[1] >= size(A')[2])
    "Need to take QR decomposition but dimensions of `A` are incorrect."

    # get thin QR
    sz = row_distribution ? size(A)[1] : size(A)[2]
    distribution = zeros(sz)
    Q1 = row_distribution ? Matrix(qr(A).Q) : Matrix(qr(A').Q)
    
    # compute distribution
    for i in 1:sz
        distribution[i] = row_distribution ? norm(view(Q1, i, :))^2 : norm(view(Q1, i, :))^2
    end

    return distribution ./ sum(distribution)
end