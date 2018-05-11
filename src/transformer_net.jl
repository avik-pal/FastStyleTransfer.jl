# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

mutable struct TransformerNet <: model
    top_layers
    residual_layers
    upsampling_layers
end

function TransformerNet()
    top_layers = Chain(ConvBlock(3=>32, (3,3), (1,1)),
                    InstanceNorm(32), x->relu.(x),
                    ConvBlock(32=>64, (3,3), (1,1)),
                    InstanceNorm(64), x->relu.(x),
                    ConvBLock(64=>128, (3,3), (1,1)),
                    InstanceNorm(128), x->relu.(x))
    residual_layers = Chain([ResidualBlock(128) for i in 1:5]...)
    upsampling_layers = Chain(UpsamplingBlock(128=>64, (3,3), (1,1), 2),
                    InstanceNorm(64),
                    UpsamplingBlock(64=>32, (3,3), (1,1), 2),
                    InstanceNorm(32),
                    ConvBlock(32=>3, (9,9), (1,1)))
    TransformerNet(top_layers, residual_layers, upsampling_layers)
end

function (t::TransformerNet)(x)
    t.upsampling_layers(t.residual_layers(t.top_layers(x)))
end
