module FastStyleTransfer

using Flux, Metalhead, Images, CuArrays, NNlib, BSON
using BSON: @save, @load
using Flux: @epochs, sub2, expand, initn, @treelike, _testmode!
using Base.Iterators: partition
using Flux.Tracker: track, data, @grad, nobacksies
using LinearAlgebra

export train, stylize

include("utils.jl")
include("layers.jl")
include("model.jl")
include("styletransfer.jl")

end # module
