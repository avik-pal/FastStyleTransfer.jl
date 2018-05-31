module FastStyleTransfer

using Flux, Metalhead
using FileIO, Images
using BSON: @save, @load
using CuArrays
using Base.Iterators: partition

export train, stylize, stylize_all

abstract type model end
abstract type layers end
abstract type block <: layers end

include("utils.jl")
include("layers.jl")
include("vgg.jl")
include("transformer_net.jl")
include("neural_style.jl")

end # module
