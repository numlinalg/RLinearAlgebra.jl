"""
    fwht!(x::AbstractVector; signs=ones(Bool, size(x)), scaling = 1)  

Performs a Fast Walsh Hadamard Transform (FWHT), modifying the vector `x`. This means that if you want an unmodified version of `x` you should copy it before calling this function. `signs` allows the user to input a boolean vector that flips the signs of the entries of the vector `x` before applying the transform. `scaling` allows the user to scale the result of the transform. Choosing a scaling of 1/sqrt{size(x)} will result in the FWHT being an orthogonal transform.

!!! Note: To avoid log computation at every call the function does not check that the dimension is a power of 2. This must be done by a separate function at an earlier point.
"""
function fwht!(x::AbstractVector; signs=ones(Bool, size(x, 1)), scaling = 1)
    ln = size(x,1)
    ls = size(signs,1)
    @assert rem(log(ln) / log(2), 1) ≈ 0 "Size of vector must be power of 2."
    # size of separation between indicies
    h = 1 
    inc = 2 * h
    total_its = div(log(ln), log(2))
    # In 1st pass scale and flip signs of entries, this does not need to be done 
    # in any other passes over the vector. To avoid unnecessary condition checking 
    # this portion of the loop has been separated out fron the others.
    for i in 1:inc:ln
        for j in i:(i+h-1)
            # Signs is vector of Bools
            z = x[j] * (signs[j] ? scaling : -scaling) 
            y = x[j+h] * (signs[j+h] ? scaling : -scaling)
            x[j] = z + y
            x[j+h] = z - y
        end
        
    end

    # Double distance between next combintation
    h *= 2
    for k = 2:total_its 
        # spacing between operations
        inc = 2 * h
        for i in 1:inc:ln
            for j in i:(i+h-1)
                # Perform fwht 
                z = x[j]  
                y = x[j+h]
                x[j] = z + y
                x[j+h] = z - y
            end
        
        end
        # Double distance between next combintation
        h *= 2
    end

end
