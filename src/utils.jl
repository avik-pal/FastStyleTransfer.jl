#----- Computational Utilities -----#

function gram_matrix(x)
    w, h, ch, b = size(x)
    T = eltype(x)
    local features = reshape(x, w*h, ch, b)
    feature_vec = [features[:,:,i] for i in 1:b]
    cat([feature_vec[i]' * feature_vec[i] for i in 1:b]..., dims = 3) / T(w * h * ch)
end

#----- Dataset Loading -----#

# NOTE: The dataset loader present here is quite naive and slow.
#       But there is a Distributed Loader in progress. So we shall
#       default to that once its ready.
