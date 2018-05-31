# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#-----------------------Utilities to load and save image---------------------------

# NOTE: The image returned is scaled to equal dimensions on both side
function load_image(filename; size::Int = -1, scale::Int = -1)
    img = load(filename)
    if(size != -1)
        img = imresize(img, (size,size))
    elseif(scale != -1)
        dims = size(img, 1)
        img = imresize(img, (dims, dims))
    end
    img = float.(channelview(img)) * 255
    # Normalize the input as per the Imagenet Data
    mean = reshape([123.68, 116.779, 103.939], (3,1,1))
    std = reshape([58.624, 57.334, 57.6], (3,1,1))
    img = (img .- mean)./std
    img = permutedims(img, [3,2,1])
    # The following line strangely throws an error
    # img = reshape(img, size(img)..., 1)
end

function save_image(filename, img, display::Bool = true)
    img = reshape(img, (size(img, 2), size(img, 1), 3))
    img = permutedims(img, [3,2,1])
    # Denormalize the data
    mean = reshape([123.68, 116.779, 103.939], (3,1,1))
    std = reshape([58.624, 57.334, 57.6], (3,1,1))
    img = img .* std .+ mean
    img = clamp.(img, 0, 255) / 255
    img = colorview(RGB{Float32}, img)
    save(filename, img)
    if(display)
        img
    end
end

function load_dataset(path, batch, total)
    z = readdir(path)
    indices = randperm(length(z))[1:total]
    paths = [joinpath(path, i) for i in z[indices]]
    images = []
    for i in paths
        push!(images, load_image(i, size = 224))
    end
    [cat(4, images[i]...) for i in partition(1:total, batch)]
end

function gram_matrix(x)
    w, h, ch, b = size(x)
    local features = reshape(x, w*h, ch, b)
    cat(3, [At_mul_B(features[:,:,i], features[:,:,i]) / (w * h * ch) for i in 1:b]...)
end

function normalize_batch(x)
    mean = reshape([123.68, 116.779, 103.939], (1,1,3,1))
    std = reshape([58.624, 57.334, 57.6], (1,1,3,1))
    x = (x .- mean) ./ std
end

#----------------------------Extension of certain functions------------------------------
using Base.std

# Not as per the defination. Just as hack to get it working
Base.std(x::TrackedArray, dim::Array; mean = Base.mean(x, dim)) =
    sqrt.(sum((x .- mean).^2, dim) ./ (prod(size(x)[dim])-1))
