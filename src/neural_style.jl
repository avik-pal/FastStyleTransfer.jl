# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

#------------------------Utilities to Train new models----------------------------

function train(train_data_path, batch_size, η, style_image_path, epochs, model_save_path, content_weight, style_weight, images = 10000)
    train_dataset = load_dataset(train_data_path, batch_size, images)
    transformer = TransformerNet() |> gpu
    optimizer = Flux.ADAM(params(transformer), η)
    style = load_image(style_image_path, size = 224)
    style = repeat(reshape(style, size(style)..., 1), outer = (1,1,1,batch_size)) |> gpu

    vgg = vgg19() |> gpu
    features_style = vgg(style)
    gram_style = [gram_matrix(y) for y in features_style];

    function loss_function(x)
        y = transformer(x)

        y = normalize_batch(y)
        # NOTE: The input to loss function is already normalized
        # x = normalize_batch(x)

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
        println("The content loss is $(content_loss) and style loss is $(style_loss) and the total loss is $(total_loss)")

        total_loss
    end

    @epochs epochs begin
        for d in train_dataset
            l = loss_function(d |> gpu);
            Flux.back!(l);
            optimizer()
        end
    end

    # NOTE: Model must be brought to device while saving since stylization might be performed on CPU
    transformer = transformer |> cpu

    @save model_save_path transformer
end

#----------------------------------Utilities to Stylize Images--------------------------------

function stylize(image_path, model_path = "../models/trained_network_1.bson"; save_path = None, display::Bool = true)
    info("Starting to Load Model")
    @load model_path style_model
    info("Model has been Loaded Successfully")
    img = load_image(image_path)
    info("Image Loaded Successfully")
    a = time()
    stylized_img = style_model(img)
    info("Image has been Stylized in $(time()-a) seconds")
    if(save_path == None)
        path = rsplit(image_path, ".", limit = 2)
        save_path = "$(path[1])_stylized.$(path[2])"
    end
    save_image(save_path, stylized_img, display)
    info("The image has been saved successfully")
end
