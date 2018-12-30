#----- Transformer Net -----#

function TransformerNet(;upsample = true, batchnorm = true)
    alias = batchnorm ? BatchNorm : InstanceNorm
    res_chain =  [ResidualBlock(128, batchnorm) for i in 1:5]
    if upsample
        model = Chain(Conv((3, 3), 3=>32, pad = (1, 1)),
                      alias(32, relu),
                      Conv((3, 3), 32=>64, stride = (2, 2), pad = (1, 1)),
                      alias(64, relu),
                      Conv((3, 3), 64=>128, stride = (2, 2), pad = (1, 1)),
                      alias(128, relu),
                      res_chain...,
                      UpsamplingBlock((3, 3), 128=>64, pad = (1, 1)),
                      alias(64),
                      UpsamplingBlock((3, 3), 64=>32, pad = (1, 1)),
                      alias(32),
                      Conv((9, 9), 32=>3))
    else
        model = Chain(Conv((3, 3), 3=>32, pad = (1, 1)),
                      alias(32, relu),
                      Conv((4, 4), 32=>64, stride = (2, 2), pad = (1, 1)),
                      alias(64, relu),
                      Conv((4, 4), 64=>128, stride = (2, 2), pad = (1, 1)),
                      alias(128, relu),
                      res_chain...,
                      ConvTranspose((4, 4), 128=>64, stride = (2, 2), pad = (1, 1)),
                      alias(64),
                      ConvTranspose((4, 4), 64=>32, stride = (2, 2), pad = (1, 1)),
                      alias(32),
                      ConvTranspose((3, 3), 32=>3, pad = (1, 1)))
    end
    return model
end

#----- Feature Extractor -----#

struct FeatureExtractor
    slices
end

@treelike FeatureExtractor

function FeatureExtractor(nslice::Int = 4, depth_each::Int = 4, model = VGG19)
    extractor = trained(model).layers
    slices = [extractor[((i - 1) * depth_each + 1):(i * depth_each)] for i in 1:nslice]
    FeatureExtractor(tuple(slices...))
end

function (f::FeatureExtractor)(x)
    output = Vector(undef, size(f.slices, 1))
    output[1] = f.slices[1](x)
    for i in 2:size(f.slices, 1)
        output[i] = f.slices[i](output[i - 1])
    end
    return output
end
