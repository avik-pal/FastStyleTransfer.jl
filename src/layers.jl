#----- Instance Normalization -----#

mutable struct InstanceNorm{F,V,W,N}
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W  # moving mean
  σ²::W  # moving var
  ϵ::N
  momentum::N
  active::Bool
end

@treelike InstanceNorm

InstanceNorm(chs::Integer, λ = identity;
             initβ = (i) -> zeros(i), initγ = (i) -> ones(i), ϵ = 1e-8, momentum = .1) =
  InstanceNorm(λ, param(initβ(chs)), param(initγ(chs)),
               zeros(chs), ones(chs), ϵ, momentum, true)

function (IN::InstanceNorm)(x)
    γ, β = BN.γ, BN.β
    dims = length(size(x))
    channels = size(x, dims-1)
    affine_shape = ones(Int, dims)
    affine_shape[end-1] = channels
    affine_shape[end] = size(x, dims)
    m = prod(size(x)[1:end-2])

    if !IN.active
        μ = reshape(IN.μ, affine_shape...)
        σ² = reshape(IN.σ², affine_shape...)
    else
        T = eltype(x)

        ϵ = data(convert(T, BN.ϵ))
        axes = [1:dims-2]
        μ = mean(x, dims = axes)
        σ² = mean((x .- μ).^2, dims = axes)

        # update moving mean/std
        mtm = data(convert(T, BN.momentum))
        IN.μ = (1 - mtm) .* IN.μ .+ mtm .* dropdims(data(μ), dims = (axes...,))
        IN.σ² = (1 - mtm) .* IN.σ² .+ mtm .* dropdims(data(σ²), dims = (axes...,)) .* m ./ (m - 1)
    end

    let λ = IN.λ
        λ.(reshape(γ, affine_shape...) .* ((x .- μ) ./ sqrt.(σ² .+ ϵ)) .+ reshape(β, affine_shape...))
    end
end

_testmode!(IN::InstanceNorm, test) = (IN.active = !test)

function Base.show(io::IO, l::InstanceNorm)
    print(io, "InstanceNorm($(join(size(l.β), ", "))")
    (l.λ == identity) || print(io, ", λ = $(l.λ)")
    print(io, ")")
end

#----- Residual Block -----#

struct ResidualBlock; layers; end

@treelike ResidualBlock

function ResidualBlock(chs::Int, batchnorm)
    alias = batchnorm ? BatchNorm : InstanceNorm
    ResidualBlock(Chain(Conv((3, 3), chs=>chs, pad=(1, 1)),
                        alias(chs, relu),
                        Conv((3, 3), chs=>chs, pad=(1, 1)),
                        alias(chs)))
end

(r::ResidualBlock)(x) = r.layers(x) .+ x

#----- Upsample -----#

Upsample(scale = 2) = x -> repeat(x, inner = (scale, scale, 1, 1))

UpsamplingBlock(kernel, chs; stride = (1, 1), scale = 2, pad = (0, 0)) =
    Chain(Conv(kernel, chs, stride = stride, pad = pad),
          Upsample(scale))

#----- Conv Transpose -----#

# NOTE: Will soon be in Flux. Remove once that PR is merged
function out_size(stride, pad, dilation, kernel, xdims)
    dims = []
    for i in zip(stride, pad, dilation, kernel, xdims)
        push!(dims, i[1] * (i[5] - 1) + (i[4] - 1) * i[3] - 2 * i[2] + 1)
    end
    dims
end

function convtranspose(x, w; stride = 1, pad = 0, dilation = 1)
    stride, pad, dilation = NNlib.padtuple(x, stride), NNlib.padtuple(x, pad), NNlib.padtuple(x, dilation)
    y = similar(x, out_size(stride, pad, dilation, size(w)[1:end-2], size(x)[1:end-2])...,size(w)[end-1],size(x)[end])
    NNlib.∇conv_data(x, y, w, stride = stride, pad = pad, dilation = dilation)
end

convtranspose(x::TrackedArray, w::TrackedArray; kw...) = track(convtranspose, x, w; kw...)
convtranspose(x::AbstractArray, w::TrackedArray; kw...) = track(convtranspose, x, w; kw...)
convtranspose(x::TrackedArray, w::AbstractArray; kw...) = track(convtranspose, x, w; kw...)

@grad convtranspose(x, w; kw...) =
    convtranspose(data.((x, w))...; kw...), Δ -> nobacksies(:convtranspose, (NNlib.conv(data.((Δ, w))...; kw...), NNlib.∇conv_filter(data.((x, Δ, w))...; kw...)))

struct ConvTranspose{N,F,A,V}
    σ::F
    weight::A
    bias::V
    stride::NTuple{N,Int}
    pad::NTuple{N,Int}
    dilation::NTuple{N,Int}
end

ConvTranspose(w::AbstractArray{T,N}, b::AbstractVector{T}, σ = identity;
    stride = 1, pad = 0, dilation = 1) where {T,N} =
    ConvTranspose(σ, w, b, expand.(sub2(Val(N)), (stride, pad, dilation))...)

ConvTranspose(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = identity; init = glorot_uniform,
    stride = 1, pad = 0, dilation = 1) where N =
    ConvTranspose(param(init(k..., ch[2], ch[1])), param(zeros(Float32, ch[2])), σ,
                  stride = stride, pad = pad, dilation = dilation)

@treelike ConvTranspose

function (c::ConvTranspose)(x)
    σ, b = c.σ, reshape(c.bias, map(_->1, c.stride)..., :, 1)
    σ.(convtranspose(x, c.weight, stride = c.stride, pad = c.pad, dilation = c.dilation) .+ b)
end

function Base.show(io::IO, l::ConvTranspose)
    print(io, "ConvTranspose(", size(l.weight)[1:ndims(l.weight)-2])
    print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end
