module FastStyleTransfer

using Flux, Metalhead, Images, CuArrays
using BSON: @save, @load
using Flux: @epochs
using Base.Iterators: partition
using Flux.Tracker: track, back, data

export train, stylize, stylize_all

include("utils.jl")
include("layers.jl")
include("vgg.jl")
include("transformer_net.jl")
include("neural_style.jl")

end # module
