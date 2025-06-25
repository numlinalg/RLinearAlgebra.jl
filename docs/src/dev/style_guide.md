# Style Guide
```@contents
Pages=["styleguide.md"]
```
When writing code for `RLinearAlgebra.jl` we expect the code to be written in accordance 
with the [BLUE](https://github.com/JuliaDiff/BlueStyle) style.

## Documentation
This section describes the writing style that should be used when writing documentation for 
`RLinearAlgebra.jl.` Many of these ideas for these suggestions
come from [JUMP](https://jump.dev/JuMP.jl/stable/developers/style/). 
Overall when documenting the code one should follow these recommendations:
    - Be concise
    - Prefer lists over long sentences
    - Use numbers when describing an ordered set of ideas
    - Use bullets when these is no specific order

### Docstrings
    - Every new **function** and **data structure** needs to have a docstring
    - Use properly punctuated complete sentences

Below, we provide an example of a function docstring and a data structure docstring.

#### Function Docstring
```
"""
    myFunction(args; kwargs...)
    
A couple of sentences describing the function. These sentences should describe what inputs 
are required and what is output by the function.

### Arguments
- `arg1`, description of arg 1

### Outputs
The result of calling the function. This should be either the data structure that is 
modified or what is returned.

A citation from the package DocumenterCitations.
"""

``` 

#### Data Structure Docstring
```
"""
    YourStructure <: YourStructuresSuperType

A brief sentence describing the purpose of the structure.

A citation in MLA format if the function comes from another author's work.

### Fields
- `S::FieldType`, brief description of field purpose

Include a sentence or two describing how the constructors work. Please be sure to include 
the default values of the constructor.
"""

``` 
