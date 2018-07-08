module FastStyleTransfer

using Flux, Metalhead, Images, CuArrays, NNlib
using BSON: @save, @load
using Flux: @epochs, sub2, expand, initn
using Base.Iterators: partition
using Flux.Tracker: track, data, @back
import Flux.Tracker.back

export train, stylize

include("utils.jl")
include("layers.jl")
include("vgg.jl")
include("transformer_net.jl")
include("neural_style.jl")

end # module
