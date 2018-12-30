#----- Trainer for Neural Style Transfer -----#

function train(dataset, style_img, image_path, save_path; batch_size = 16, η = 0.001, epochs = 10,
               model_save_path = "./styletransfer.bson", content_weight = 10, style_weight = 2, depth = 3)

    try
        @load model_save_path transformer
    catch
        transformer = TransformerNet(upsample = false)
    end
    transformer = transformer |> gpu
    ps = params(transformer)
    optimizer = Flux.ADAM(η)
    extractor = FeatureExtractor() |> gpu
    style = StyleImage(style_img, extractor, batch_size)
    train_data = COCODataset(dataset, batch_size)

    function loss_function(y, x, depth::Int = depth)
        features_x = extractor(x)
        features_y = extractor(y)

        content_loss = content_weight * Flux.mse(features_x[depth], features_y[depth])

        style_loss = 0.0
        for i in 1:length(style.gram_matrix)
            style_loss += Flux.mse(style.gram_matrix[i], gram_matrix(features_y[i]))
        end
        style_loss *= style_weight

        push!(style_loss_arr, data(style_loss |> cpu))
        push!(content_loss_arr, data(content_loss |> cpu))

        total_loss = style_loss + content_loss

        return total_loss
    end

    function stylize_image()
        Flux.testmode!(transformer)

        img = load_image(img_path) |> gpu

        stylized_img = transformer(img)
        save_image(save_path, data(stylized_img |> cpu), display_img)

        Flux.testmode!(transformer, false)
    end

    while train_data.complete != true
        train_dataset = train_data()
        style_loss_arr = []
        content_loss_arr = []
        @epochs epochs begin
            for d in train_dataset
                size(d, 4) != batch_size && continue # Needed as we are using batch size in StyleImage
                d = d |> gpu
                l = loss_function(transformer(d), d)
                Flux.back!(l)
                Flux.Optimise.update!(optimizer, ps)
            end
            stylize_image()
            println("Running Style Loss : $(mean(style_loss_arr))")
            println("Running Content Loss : $(mean(content_loss_arr))")
        end
        transformer = transformer |> cpu
        @save model_save_path transformer
        transformer = transformer |> gpu
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
