#----- Computational Utilities -----#

using Flux.Tracker: TrackedArray, track, @grad

function gram_matrix(x)
    w, h, ch, b = size(x)
    T = eltype(data(x))
    features = reshape(x, w*h, ch, b)
    return gram_matrix_fast(features)
end

gram_matrix_fast(x::TrackedArray) = track(gram_matrix_fast, x)

gram_matrix_fast(x::CuArray) = CuArrays.CUBLAS.gemm_strided_batched('T', 'N', x, x) / prod(size(x)[1:2])

gram_matrix_fast(x) = cat([x[:, :, i]' * x[:, :, i] for i in 1:size(x, 3)]..., dims = 3) / prod(size(x)[1:2])

∇gram_matrix_fast(x::CuArray, Δ::CuArray) =
    CuArrays.CUBLAS.gemm_strided_batched('N', 'N', x, Δ .+ Δ') * prod(size(x)[1:2])

function ∇gram_matrix_fast(x, Δ)
    Δ₂ = Δ .+ Δ'
    cat([x[:, :, i] * Δ₂[:, :, i] for i in 1:size(x, 3)]..., dims = 3) * prod(size(x)[1:2])
end

@grad function gram_matrix_fast(x)
    return gram_matrix_fast(data(x)), Δ -> (∇gram_matrix_fast(data(x), Δ), )
end

# Not consistent with the definition but gives proper outcome

function gram_matrix_v2(x)
    h, w, c, b = size(x)
    x = reshape(x, h * w, c * b)
    return (x' * x) / (h * w * c * b)
end

#----- Image Manipulation -----#

function load_image(filename; size_img::Int = -1)
    im = load(filename)
    global original_size = size(im)
    size_img != -1 && (im = imresize(im, (size_img,size_img)))
    im = channelview(im)
    μ = [0.485, 0.456, 0.406]
    σ = [0.229, 0.224, 0.225]
    ndims(im) != 3 && return im
    im = (im .- μ)./σ
    return Float32.(permutedims(im, (3, 2, 1))[:,:,:,:].*255)
end

function save_image(filename, img, display_img::Bool = false)
    img = dropdims(img, dims = 4)
    img = permutedims(img, [3,2,1])
    μ = [0.485, 0.456, 0.406]
    σ = [0.229, 0.224, 0.225]
    img = ((img ./ 255) .* σ) .+ μ
    clamp!(img, 0.0, 1.0)
    img = colorview(RGB{eltype(img)}, img)
    img = imresize(img, original_size)
    display_img && display(img)
    save(filename, img)
end

function load_dataset(c::COCODataset, size_img = 224)
    images = Vector(undef, c.max_load)
    loaded = 0
    while true
        c.last_loaded += 1
        if c.last_loaded > length(c.file_paths)
            c.last_loaded = 1
        end
        img = load_image(c.file_paths[c.last_loaded], size_img = size_img)
        if ndims(img) == 3
            images[loaded + 1] = img
            loaded += 1
        end
        loaded == c.max_load && break
    end
    return [cat(images[i]..., dims = 4) for i in partition(1:loaded, c.batch_size)]
end

#----- Dataset Loading -----#

# NOTE: The dataset loader present here is quite naive and slow.
#       But there is a Distributed Loader in progress. So we shall
#       default to that once its ready.

mutable struct COCODataset
    file_paths
    last_loaded
    batch_size
    complete
    max_load
end

function COCODataset(dataset, batch_size = 16, max_load = 16 * 100)
    files = (dataset * "/") .* readdir(dataset)
    COCODataset(files, 1, batch_size, false, max_load)
end

function (c::COCODataset)(;size_img = 224)
    images = []
    @showprogress 0.1 "Loading Images..." for i in 1:c.max_load
        img = load_image(c.file_paths[c.last_loaded], size_img = size_img)
        ndims(img) == 4 && push!(images, img) # Hack to avoid errors in case of MSCOCO
        c.last_loaded = (c.last_loaded + 1) % length(c.file_paths)
        c.last_loaded == 0 && (c.complete = true; c.last_loaded = 1)
    end
    return [cat(images[i]..., dims = 4) for i in partition(1:length(images), c.batch_size)]
end

#----- Style Image Descriptor -----#

struct StyleImage
    img_path
    features
    gram_style
end

function StyleImage(filename, extractor, batch_size::Int, size_img = 224)
    img = load_image(filename, size_img = size_img)
    img = repeat(img, outer = (1, 1, 1, batch_size)) |> gpu
    features = extractor(img)
    gram_style = [gram_matrix(x) for x in features]
    StyleImage(filename, features, gram_style)
end
