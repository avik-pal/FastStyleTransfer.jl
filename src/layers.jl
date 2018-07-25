# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#-------------------Instance Normalization-----------------------------------

# NOTE: The Instance Normalization code is slow and can act as a huge bottleneck.
# Hence until this issue is fixed we shall be using BatchNorm

struct InstanceNorm
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

struct ResidualBlock
    conv_layers
    norm_layers
end

Flux.treelike(ResidualBlock)

ResidualBlock(chs::Int) =
   ResidualBlock((Conv((3,3), chs=>chs, pad = (1,1)), Conv((3,3), chs=>chs, pad = (1,1))), (BatchNorm(chs), BatchNorm(chs)))

function (r::ResidualBlock)(x)
    value = relu.(r.norm_layers[1](r.conv_layers[1](x)))
    r.norm_layers[2](r.conv_layers[2](value)) + x
end

#--------------------------Reflection Pad-------------------------------------

# Paper suggests using Reflection Padding. However normal padding is being used until this layer is implemented

struct ReflectionPad
    dim::Int
end

Flux.treelike(ReflectionPad)

#----------------------Convolution Block--------------------------------------

ConvBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int} = (1,1), pad::Tuple{Int,Int} = (0,0)) =
    Chain(Conv(kernel, chs, stride = stride, pad = pad), ReflectionPad(kernel[1]÷2))

#-------------------------Upsample--------------------------------------------

Upsample(x) = repeat(x, inner = (2,2,1,1))

#----------------------Upsampling BLock---------------------------------------

# TODO: Use reflection padding instead of zero padding once its implemented

UpsamplingBlock(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int}, upsample::Int, pad::Tuple{Int,Int} = (0,0)) =
    Chain(Conv(kernel, chs, stride = stride, pad = (kernel[1]÷2, kernel[2]÷2)), x -> Upsample(x))

#---------------------Convolution Transpose-----------------------------------

function out_size(stride, pad, dilation, kernel, xdims, output_pad)
    dims = []
    for i in zip(stride, pad, dilation, kernel, xdims, output_pad)
        push!(dims, i[1] * (i[5] - 1) + (i[4] - 1) * i[3] - 2 * i[2] + 1 + i[6])
    end
    dims
end

function convtranspose(x, w; stride = 1, pad = 0, dilation = 1, output_pad = 0)
    stride, pad, dilation, output_pad = NNlib.padtuple(x, stride), NNlib.padtuple(x, pad), NNlib.padtuple(x, dilation), NNlib.padtuple(x, output_pad)
    y = similar(x, out_size(stride, pad, dilation, size(w)[1:end-2], size(x)[1:end-2], output_pad)...,size(w)[end-1],size(x)[end])
    NNlib.∇conv_data(x, y, w, stride = stride, pad = pad, dilation = dilation)
end

convtranspose(x::TrackedArray, w::TrackedArray; kw...) = track(convtranspose, x, w; kw...)
convtranspose(x::AbstractArray, w::TrackedArray; kw...) = track(convtranspose, x, w; kw...)
convtranspose(x::TrackedArray, w::AbstractArray; kw...) = track(convtranspose, x, w; kw...)

@grad convtranspose(x, w; kw...) =
    convtranspose(data.((x, w))...; kw...), Δ -> nobacksies(:convtranspose, (NNlib.conv(data.((Δ, w))...; kw[1:end-1]...), NNlib.∇conv_filter(data.((x, Δ, w))...; kw[1:end-1]...)))

struct ConvTranspose{N,F,A,V}
    σ::F
    weight::A
    bias::V
    stride::NTuple{N,Int}
    pad::NTuple{N,Int}
    dilation::NTuple{N,Int}
    output_pad::NTuple{N,Int}
end

ConvTranspose(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
    stride = 1, pad = 0, dilation = 1, output_pad = 0) where {T,N} =
    ConvTranspose(σ, w, b, expand.(sub2(Val{N}), (stride, pad, dilation, output_pad))...)

ConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; init = Flux.initn,
    stride = 1, pad = 0, dilation = 1, output_pad = 0) where N =
    ConvTranspose(param(init(k..., ch[2], ch[1])), param(zeros(ch[2])), σ,
                  stride = stride, pad = pad, dilation = dilation, output_pad = output_pad)

Flux.treelike(ConvTranspose)

function (c::ConvTranspose)(x)
    σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
    σ.(convtranspose(x, c.weight, stride = c.stride, pad = c.pad, dilation = c.dilation, output_pad = c.output_pad) .+ b)
end

function Base.show(io::IO, l::ConvTranspose)
    print(io, "ConvTranspose(", size(l.weight)[1:ndims(l.weight)-2])
    print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end
