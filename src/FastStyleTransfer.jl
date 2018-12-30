module FastStyleTransfer

using Flux, Metalhead, Images, CuArrays, NNlib, BSON
using BSON: @save, @load
using Flux: @epochs, sub2, expand, glorot_uniform, @treelike
import Flux._testmode!
using Base.Iterators: partition
using Flux.Tracker: track, data, @grad, nobacksies
using Statistics
using ProgressMeter

export train, stylize

include("utils.jl")
include("layers.jl")
include("model.jl")
include("styletransfer.jl")

end # module
