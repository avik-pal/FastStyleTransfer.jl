# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#-----------------------------Transformer Net--------------------------------------

# Replace Instance Norm with Batch Norm until GPU speed issue is fixed

ConvPad(chs::Pair{<:Int,<:Int}, kernel::Tuple{Int,Int}, stride::Tuple{Int,Int} = (1,1)) =
    Conv(kernel, chs, stride = stride, pad = (kernel[1]÷2, kernel[2]÷2))

TransformerNet() = Chain(ConvPad(3=>32, (3,3)),
                         BatchNorm(32, relu),
                         ConvPad(32=>64, (3,3), (2,2)),
                         BatchNorm(64, relu),
                         ConvPad(64=>128, (3,3), (2,2)),
                         BatchNorm(128, relu),
                         [ResidualBlock(128) for i in 1:5]...,
                         UpsamplingBlock(128=>64, (3,3), (1,1), 2),
                         BatchNorm(64),
                         UpsamplingBlock(64=>32, (3,3), (1,1), 2),
                         BatchNorm(32),
                         ConvPad(32=>3, (9,9), (1,1)))

TransformerNet2() = Chain(ConvPad(3=>32, (3,3)),
                          BatchNorm(32, relu),
                          ConvPad(32=>64, (3,3), (2,2)),
                          BatchNorm(64, relu),
                          ConvPad(64=>128, (3,3), (2,2)),
                          BatchNorm(128, relu),
                          [ResidualBlock(128) for i in 1:5]...,
                          ConvTranspose((3, 3), 128=>64, stride = (2, 2), pad = (1, 1), output_size = (112, 112)),
                          BatchNorm(64, relu),
                          ConvTranspose((3, 3), 64=>32, stride = (2, 2), pad = (1, 1), output_size = (224, 224)),
                          BatchNorm(32, relu),
                          ConvTranspose((3, 3), 32=>3, pad = (1, 1), output_size = (224, 224)))
