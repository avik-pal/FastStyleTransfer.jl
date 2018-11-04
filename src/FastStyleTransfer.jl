module FastStyleTransfer

using Flux, Metalhead, Images, CuArrays, NNlib, BSON
using BSON: @save, @load
using Flux: @epochs, sub2, expand, initn, @treelike, _testmode!
using Base.Iterators: partition
using Flux.Tracker: track, data, @grad, nobacksies

export train, stylize

include("utils.jl")
include("layers.jl")
include("vgg.jl")
include("transformer_net.jl")
include("neural_style.jl")

end # module
