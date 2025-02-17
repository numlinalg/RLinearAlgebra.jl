# This file is part of RLinearAlgebra.jl

# Function collect_samplers is defined to collect all the linear 
# samplers defined in the package, which is used for testing.

function collect_samplers(collect_type::String)
     sampler_prefixes = Dict(
        "vec" => "LinSysVec",
        "blk" => "LinSysBlk"
     )

     sampler_types = Type[]
     # Match types
     prefix = get(sampler_prefixes, collect_type, nothing)

     if prefix === nothing
          @warn "Unknown collect_type: $collect_type"
          return sampler_types
     end


     for sym in names(RLinearAlgebra; all=false)
          # Skip non datatype ones
          if isdefined(RLinearAlgebra, sym)
               T = getproperty(RLinearAlgebra, sym)

               if !(T isa DataType)
                    continue
               end
          else 
               continue
          end
          

          if startswith(string(T), prefix) && T <: Union{eval(Symbol(prefix * "RowSampler")), eval(Symbol(prefix * "RowSelect")), eval(Symbol(prefix * "ColSampler")), eval(Symbol(prefix * "ColSelect"))}
               # Check whether we can call with no arguments
               if hasmethod(T, ())
                   push!(sampler_types, T)
               else
                   continue
               end
           end
     end
     return sampler_types
end
