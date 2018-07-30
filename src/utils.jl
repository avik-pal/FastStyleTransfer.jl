# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#-----------------------Utilities to load and save image---------------------------

im_mean = reshape([0.485, 0.458, 0.408], (1,1,3)) * 255
im_mean2 = reshape([0.485, 0.458, 0.408], (1,1,3,1)) * 255 |> gpu

# NOTE: The image returned is scaled to equal dimensions on both side
function load_image(filename; size_img::Int = -1, scale::Int = -1)
    img = load(filename)
    global original_size = size(img)
    if size != -1
        img = imresize(img, (size_img,size_img))
    elseif scale != -1
        dims = size(img, 1)
        img = imresize(img, (dims, dims))
    end
    img = Float32.(channelview(img)) * 255
    ndims(img) == 2 && return img
    permutedims(img, [3,2,1]) .- im_mean
end

function save_image(filename, img, display_img::Bool = false)
    img = reshape(img, (size(img, 2), size(img, 1), 3))
    img = permutedims(img, [3,2,1])
    img = (img .+ im_mean) / 255
    img -= minimum(img)
    img /= maximum(img)
    img = colorview(RGB{eltype(img)}, img)
    img = imresize(img, original_size)
    display_img && display(img)
    save(filename, img)
end

function load_dataset(path, batch, total)
    z = readdir(path)
    indices = randperm(length(z))[1:total]
    paths = [joinpath(path, i) for i in z[indices]]
    images = []
    for (counts, i) in enumerate(paths)
        img = load_image(i, size_img = 224)
        ndims(img) == 3 ? push!(images, img) : total -= 1 # Hack to avoid errors in case of MSCOCO
        counts % 100 == 0 && info("$counts images have been loaded")
    end
    [cat(4, images[i]...) for i in partition(1:total, batch)]
end

# NOTE: The wrapper is quite slow
# function gram_matrix(x::CuArray)
#     w, h, ch, b = size(x)
#     local features = reshape(x, w*h, ch, b)
#     features = [features[:,:,i] for i in 1:b]
#     res = CuArrays.BLAS.gemm_batched('T', 'N', features, features)
#     cat(3, res...) / (w * h * ch)
# end

function gram_matrix(x)
    w, h, ch, b = size(x)
    local features = reshape(x, w*h, ch, b)
    features = [features[:,:,i] for i in 1:b]
    cat(3, [features[i]' * features[i] for i in 1:b]...) / Float32(w * h * ch)
end

#----------------------------Extension of certain functions------------------------------
using Base.std

# Not as per the defination. Just as hack to get it working
Base.std(x::TrackedArray, dim::Array; mean = Base.mean(x, dim)) =
    sqrt.(sum((x .- mean).^2, dim) ./ (prod(size(x)[dim])-1))
