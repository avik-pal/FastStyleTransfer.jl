# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#-------------------Instance Normalization-----------------------------------

# NOTE: The Instance Normalization code is slow and can act as a huge bottleneck.
# Hence until this issue is fixed we shall be using BatchNorm

mutable struct InstanceNorm <: layers
    β
    γ
end

Flux.treelike(InstanceNorm)

InstanceNorm(chs::Int; initβ = zeros, initγ = ones) = InstanceNorm(param(initβ(chs)), param(initγ(chs)))

function (IN::InstanceNorm)(x)
    local chs = length(IN.β.data)
    reshape(IN.γ, (1,1,chs,1)) .* ((x .- mean(x, [1,2,3])) ./ std(x, [1,2,3])) .+ reshape(IN.β, (1,1,chs,1))
end

#---------------------------Residual Block-----------------------------------

mutable struct ResidualBlock <: block
    conv_layers
    norm_layers
end

Flux.treelike(ResidualBlock)

function ResidualBlock(chs::Int)
    ResidualBlock((Conv((3,3), chs=>chs, pad = (1,1)), Conv((3,3), chs=>chs, pad = (1,1))), (InstanceNorm(chs), InstanceNorm(chs)))
end

function (r::ResidualBlock)(x)
    value = relu.(r.norm_layers[1](r.conv_layers[1](x)))
    r.norm_layers[2](r.conv_layers[2](value)) + x
end

#--------------------------Reflection Pad-------------------------------------

# Paper suggests using Reflection Padding. However normal padding is being used until this layer is implemented

mutable struct ReflectionPad <: layers
    dim::Int
end

Flux.treelike(ReflectionPad)

#----------------------Convolution Block--------------------------------------

mutable struct ConvBlock <: block
    pad
    conv
end

Flux.treelike(ConvBlock)

function ConvBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int} = (1,1), pad::Tuple{Int,Int} = (0,0))
    ConvBlock(ReflectionPad(kernel[1]÷2), Conv(kernel, chs, stride = stride, pad = pad))
end

(c::ConvBlock)(x) = c.conv(c.pad(x))

#-------------------------Upsample--------------------------------------------

# NOTE: The following code is implemented by @staticfloat and is currently an unmerged PR in Flux.jl.

_repeat(A, inner, outer) = Base.repeat(A; inner=inner, outer=outer)
Base.repeat(A::TrackedArray; inner=ntuple(x->1, ndims(A)), outer=ntuple(x->1, ndims(A))) = track(_repeat, A, inner, outer)

function back(::typeof(_repeat), Δ, xs::TrackedArray, inner, outer)
    Δ′ = similar(xs.data)
    Δ′ .= 0
    S = size(xs.data)

    # Loop through each element of Δ, calculate source dimensions, accumulate into Δ′
    for (dest_idx, val) in enumerate(IndexCartesian(), Δ)
        # First, round dest_idx[dim] to nearest gridpoint defined by inner[dim], then
        # wrap around based on original size S.
        src_idx = [mod1(div(dest_idx[dim] - 1, inner[dim]) + 1, S[dim]) for dim in 1:length(S)]
        Δ′[src_idx...] += val
    end
    back(xs, Δ′)
end

Upsample(x) = repeat(x, inner = (2,2,1,1))

# Upsample(x::TrackedArray) = Tracker.track(Upsample, x)

# Tracker.back(::typeof(Upsample), Δ, x) = Tracker.@back(x, maxpool(Δ, (2,2), stride = (2,2)))

#----------------------Upsampling BLock---------------------------------------

# TODO: Use reflection padding instead of zero padding once its implemented

# mutable struct UpsamplingBlock <: block
#     upsample
#     pad
#     conv
# end

# function (u::UpsamplingBlock)(x)
#     u.conv(u.pad(u.upsample(x)))
# end

# UpsamplingBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int}, upsample::Int, pad::Tuple{Int,Int} = (0,0)) = UpsamplingBlock(x -> Upsample(x), ReflectionPad(kernel[1]÷2), Conv(kernel, chs, stride = stride, pad = pad))

mutable struct UpsamplingBlock <: block
    upsample
    conv
end

Flux.treelike(UpsamplingBlock)

UpsamplingBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int}, upsample::Int, pad::Tuple{Int,Int} = (0,0)) = UpsamplingBlock(x -> Upsample(x), Conv(kernel, chs, stride = stride, pad = (kernel[1]÷2, kernel[2]÷2)))

(u::UpsamplingBlock)(x) = u.conv(u.upsample(x))
