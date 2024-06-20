############################################################################################
## This file is part of RLinearAlgebra.jl
##
## Overview: abstractions and methods for obtaining a low rank approximation of a matrix 
##
## Contents
## - Abstract Types
## - `approximate` function 
## - General Decompostion
## - Symmetric Decomposition
##
############################################################################################

# Dependencies: Random, Distributions, LinearAlgebra

#############################################
# Abstract Types
#############################################

# Here is the skeleton for the    


"""ApproxMethod 

Abstract supertype for low rank approximations to matrices. 

"""
abstract type ApproxMethod end

"""
    RangeFinderMethod <: ApproxMethod 

Abstract supertype for Random Rangefinder techniques such as the random svd decomposition
random qr decomposition, and the random eigen decompositions.
"""
abstract type RangeFinderMethod <: ApproxMethod end

"""
     IntDecompMethod <: ApproxMethod 

Abstract supertype for Interpolatory decompositions. This includes methods like the CUR 
decomposition.
"""
abstract type IntDecompMethod <: ApproxMethod end

"""
    NystromMethod <: ApproxMethod

Abstract supertype for Nystrom techniques. These can only be applied to symmetric matrices.
"""
abstract type NystromMethod <: ApproxMethod end

"""
    approximate(type::T where T<:ApproxMethod,
                A::AbstractArray
               )
A common interface for specifying different strategies for forming low rank approximations
of matrices using randomized techniques. This function edits the method data structure in place
but will also return the decomposition as well as an error metric representing the 
decompositions quality, if an error technique is specified.
"""
function approximate(
    type::ApproxMethod,
    A::AbstractArray,
)
    return nothing
end

#############################################
# General Decompositions
#############################################

#############################################
# Symmetric Decompositions
#############################################


#############################################
# Export Statements
#############################################
