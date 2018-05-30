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
    mean = [123.68, 116.779, 103.939]
    std = [58.624, 57.334, 57.6]
    for i in 1:3
        img[i,:,:] = (img[i,:,:] - mean[i]) / std[i]
    end
    img = permutedims(img, [2,3,1])
    # The following line strangely throws an error
    # img = reshape(img, size(img)..., 1)
end

function save_image(filename, img, display::Bool = true)
    img = reshape(img, (size(img, 2), size(img, 1), 3))
    img = permutedims(img, [3,1,2])
    # Denormalize the data
    mean = [123.68, 116.779, 103.939]
    std = [58.624, 57.334, 57.6]
    for i in 1:3
        img[i,:,:] = img[i,:,:] * std[i] + mean[i]
    end
    img = clamp.(img, 0, 255) / 255
    img = colorview(RGB{Float32}, img)
    save(filename, img)
    if(display)
        img
    end
end

function load_dataset()
end

function gram_matrix(x)
    w, h, ch, b = size(x)
    local features = reshape(x, w*h, ch, b)
    cat(3, [At_mul_B(features[:,:,i], features[:,:,i]) / (w * h * ch) for i in 1:b]...)
end

function normalize_batch(x)
    mean = [123.68, 116.779, 103.939]
    std = [58.624, 57.334, 57.6]
    for i in 1:3
        x[:,:,i,:] = (x[:,:,i,:] - mean[i]) ./ std[i]
    end
    x
end

#----------------------------Extension of certain functions------------------------------
using Base.std

# Not as per the defination
Base.std(x::TrackedArray, dim::Array; mean = Base.mean(x, dim)) =
    sqrt.(sum((x .- mean).^2, dim) ./ (prod(size(x)[dim])-1))
