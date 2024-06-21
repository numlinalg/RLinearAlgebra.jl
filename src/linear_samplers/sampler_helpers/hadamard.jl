"""
fwht!(x::AbstractVector; signs=ones(Bool, size(x)), scaling = 1)  

Performs an in-place Fast Walsh Hadamard Transform (FWHT). `signs` allows the user to input a booleanvector that can flip the signs in the transform. `scaling` allows the user to scale the result of the transform. 

!!! Note: To avoid log computation at every call the function does not check that the dimension is a power of 2. This must be done by a separate function at an earlier point.
"""
function fwht!(x::AbstractVector; signs=ones(Bool, size(x)), scaling = 1)
    ln = size(x,1)
    ls = size(signs,1)
    # size of separation between indicies
    h = 1 
    inc = 2 * h
    # In 1st pass scale and flip signs of entries
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
    while h < ln
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
