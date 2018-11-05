function train()
    function loss_function(y, x, style::StyleImage, depth::Int = 3)
        features_x = extractor(x)
        features_y = extractor(y)

        content_loss = content_weight * Flux.mse(features_x[depth], features_y[depth])
        style_loss = style_weight * sum(Flux.mse.(style.gram_style, gram_matrix.(features_y)))

        total_loss = style_loss + content_loss

        return total_loss
    end

    function stylize_image(img_path, save_path)
        Flux.testmode!(transformer)

        img = load_image(img_path)
        img = reshape(img, size(img)..., 1) |> gpu

        stylized_img = transformer(img)
        save_image(save_path, data(stylized_img |> cpu), display_img)

        Flux.testmode!(transformer, false)
    end
end

#----- Stylize -----#

function stylize(image_path, model_path; save_path = "", display_img::Bool = true)
    @load model_path transformer
    transformer = transformer |> gpu
    Flux.testmode!(transformer)
    img = load_image(image_path)
    img = reshape(img, size(img)..., 1) |> gpu
    a = time()
    stylized_img = transformer(img)
    info("$(size(img, 1) × size(img, 2)) Image Stylized in $(time()-a) seconds")
    if(save_path == "")
        path = rsplit(image_path, ".", limit = 2)
        save_path = "$(path[1])_stylized.$(path[2])"
    end
    stylized_img = stylized_img |> cpu
    save_image(save_path, data(stylized_img), display_img)
    @info "Stylized Image saved at $(save_path)"
end
