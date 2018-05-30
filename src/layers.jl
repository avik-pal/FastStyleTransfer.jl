# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#-------------------Instance Normalization-----------------------------------

mutable struct InstanceNorm <: layers
    β
    γ
    ϵ
end

Flux.treelike(InstanceNorm)

InstanceNorm(chs::Int; initβ = zeros, initγ = ones, ϵ = 1.0e-8) = InstanceNorm(param(initβ(chs)), param(initγ(chs)), ϵ)

# NOTE: Calculating the std on cpu is much faster
function (IN::InstanceNorm)(x)
    local chs = length(IN.β.data)
    ẋ = reshape(x, :, size(x,4))
    reshape(IN.γ, (1,1,chs,1)) .* ((x .- mean(ẋ, 1)) ./ std(ẋ, 1)) .+ reshape(IN.β, (1,1,chs,1))
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

# mutable struct ReflectionPad <: layers
#     dim::Int
# end

# Flux.treelike(ReflectionPad)

#----------------------Convolution Block--------------------------------------

mutable struct ConvBlock <: block
    # pad
    conv
end

Flux.treelike(ConvBlock)

function ConvBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int} = (1,1), pad::Tuple{Int,Int} = (0,0))
    # ConvBlock(ReflectionPad(kernel[1]÷2), Conv(kernel, chs, stride = stride, pad = pad))
    ConvBlock(Conv(kernel, chs, stride = stride, pad = (kernel[1]÷2, kernel[2]÷2)))
end

# (c::ConvBlock)(x) = c.conv(c.pad(x))
(c::ConvBlock)(x) = c.conv(x)

#-------------------------Upsample--------------------------------------------

# NOTE: There is an unmerged PR in Flux.jl implementing repeat for tracked arrays. Remove this code block once it is merged

Upsample(x) = repeat(x, inner = (2,2,1,1))

Upsample(x::TrackedArray) = Tracker.track(Upsample, x)

Tracker.back(::typeof(Upsample), Δ, x) = Tracker.@back(x, maxpool(Δ, (2,2), stride = (2,2)))

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
