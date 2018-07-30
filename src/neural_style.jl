# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#------------------------Utilities to Train new models----------------------------

function train(train_data_path, batch_size, η, style_image_path, epochs, model_save_path, content_weight, style_weight, model = TransformerNet; images = 10000)
    train_dataset = load_dataset(train_data_path, batch_size, images)
    try
        @load model_save_path transformer
    catch
        transformer = model()
    end
    transformer = transformer |> gpu
    optimizer = Flux.ADAM(params(transformer), η)
    style = load_image(style_image_path, size_img = 224)
    style = repeat(reshape(style, size(style)..., 1), outer = (1,1,1,batch_size)) |> gpu

    vgg = vgg19() |> gpu
    features_style = vgg(style)
    gram_style = [gram_matrix(y) for y in features_style]

    function loss_function(x)
        y = transformer(x)

        y = y .- im_mean2
        x = x .- im_mean2

        features_y = vgg(y)
        features_x = vgg(x)

        # TODO: Train models for other depths by changing the index number
        content_loss = content_weight * Flux.mse(features_y[2], features_x[2])

        style_loss = 0.0
        for i in 1:size(features_style, 1)
            gram_sty = gram_style[i]
            gram_y = gram_matrix(features_y[i])
            style_loss = style_loss + Flux.mse(gram_y, gram_sty)
        end
        style_loss = style_loss * style_weight

        total_loss = content_loss + style_loss
        println("Content Loss : $(content_loss) || Style Loss : $(style_loss) || Total Loss : $(total_loss)")

        total_loss
    end

    @epochs epochs begin
        for d in train_dataset
            size(d, 4) != batch_size && continue
            l = loss_function(d |> gpu);
            Flux.back!(l);
            optimizer()
        end
        transformer = transformer |> cpu
        @save model_save_path transformer
        transformer = transformer |> gpu
        optimizer = Flux.ADAM(params(transformer), η)
    end
end

#----------------------------------Utilities to Stylize Images--------------------------------

function stylize(image_path, model_path = "../models/trained_network_1.bson"; save_path = "", display_img::Bool = true)
    info("Starting to Load Model")
    @load model_path transformer
    transformer = transformer |> gpu
    Flux.testmode!(transformer)
    info("Model has been Loaded Successfully")
    img = load_image(image_path)
    img = reshape(img, size(img)..., 1) |> gpu
    info("Image Loaded Successfully")
    a = time()
    stylized_img = transformer(img)
    info("Image has been Stylized in $(time()-a) seconds")
    if(save_path == "")
        path = rsplit(image_path, ".", limit = 2)
        save_path = "$(path[1])_stylized.$(path[2])"
    end
    stylized_img = stylized_img |> cpu
    save_image(save_path, data(stylized_img), display_img)
    info("The image has been saved successfully")
end
