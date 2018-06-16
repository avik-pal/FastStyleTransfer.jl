# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#-----------------------Utilities to load and save image---------------------------

im_mean = reshape([0.485, 0.458, 0.408], (1,1,3))
im_std = reshape([0.261, 0.255, 0.256], (1,1,3))
im_mean2 = reshape([0.485, 0.458, 0.408], (1,1,3,1)) |> gpu
im_std2 = reshape([0.261, 0.255, 0.256], (1,1,3,1)) |> gpu

# NOTE: The image returned is scaled to equal dimensions on both side
function load_image(filename; size::Int = -1, scale::Int = -1)
    img = load(filename)
    if(size != -1)
        img = imresize(img, (size,size))
    elseif(scale != -1)
        dims = size(img, 1)
        img = imresize(img, (dims, dims))
    end
    img = Float32.(channelview(img))
    # Normalize the input as per the Imagenet Data
    if(ndims(img) == 2)
        return img
    end
    (permutedims(img, [3,2,1]) .- im_mean)./im_std
    # The following line strangely throws an error
    # img = reshape(img, size(img)..., 1)
end

function save_image(filename, img)
    img = reshape(img, (size(img, 2), size(img, 1), 3))
    img = permutedims(img, [3,2,1])
    # Denormalize the data
    img = img .* im_std .+ im_mean
    img = clamp.(img, 0, 1)
    img = colorview(RGB{eltype(img)}, img)
    save(filename, img)
end

function load_dataset(path, batch, total)
    z = readdir(path)
    indices = randperm(length(z))[1:total]
    paths = [joinpath(path, i) for i in z[indices]]
    images = []
    for i in paths
        img = load_image(i, size = 224)
        if(ndims(img) == 3) # Hack to avoid errors in case of MSCOCO
            push!(images, img)
        else
            total -= 1
        end
    end
    [cat(4, images[i]...) for i in partition(1:total, batch)]
end

function gram_matrix(x)
    w, h, ch, b = size(x)
    local features = reshape(x, w*h, ch, b)
    cat(3, [features[:,:,i]' * features[:,:,i] / (w * h * ch) for i in 1:b]...)
end

normalize_batch(x) =  (x .- im_mean2) ./ im_std2

#----------------------------Extension of certain functions------------------------------
using Base.std

# Not as per the defination. Just as hack to get it working
Base.std(x::TrackedArray, dim::Array; mean = Base.mean(x, dim)) =
    sqrt.(sum((x .- mean).^2, dim) ./ (prod(size(x)[dim])-1))
