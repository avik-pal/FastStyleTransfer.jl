#----- Computational Utilities -----#

function gram_matrix(x)
    w, h, ch, b = size(x)
    T = eltype(x)
    local features = reshape(x, w*h, ch, b)
    feature_vec = [features[:,:,i] for i in 1:b]
    cat([feature_vec[i]' * feature_vec[i] for i in 1:b]..., dims = 3) / T(w * h * ch)
end

#----- Image Manipulation -----#

function load_image(filename; size_img::Int = -1)
    im = load(filename)
    global original_size = size(img)
    size_img != -1 && (im = imresize(im, (size_img,size_img))
    μ = [0.485, 0.456, 0.406]
    σ = [0.229, 0.224, 0.225]
    im = (channelview(im) .- μ)./σ
    return Float32.(permutedims(im, (3, 2, 1))[:,:,:,:].*255)
end

function save_image(filename, img, display_img::Bool = false)
    img = dropdims(img, dims = 4)
    μ = [0.485, 0.456, 0.406]
    σ = [0.229, 0.224, 0.225]
    img = ((img ./ 255) .* σ) .+ μ
    img = permutedims(img, [3,2,1])
    img -= minimum(img)
    img /= maximum(img)
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
    [cat(images[i]..., dims = 4) for i in partition(1:loaded, c.batch_size)]
end

#----- Dataset Loading -----#

# NOTE: The dataset loader present here is quite naive and slow.
#       But there is a Distributed Loader in progress. So we shall
#       default to that once its ready.

struct COCODataset
    file_paths
    last_loaded
    batch_size
    max_load
end

function COCODataset(dataset, batch_size = 16, max_load = 16 * 100)
    files = dataset .* readdir(dataset)
    COCODataset(files, 0, batch_size, max_load)
end

#----- Style Image Descriptor -----#

struct StyleImage
    img_path
    model_save_path
    features
    gram_style
end

function StyleImage(filename, extractor, size_img = 224, model_save_path = "./model.bson")
    img = load_image(filename, size_img = size_img)
    img = reshape(img, (:, :, :, 1)) |> gpu
    features = extractor(img)
    gram_style = [gram_matrix(x) for x in features]
    StyleImage(filename, model_save_path, features, gram_style)
end
