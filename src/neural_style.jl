# The license for this code is available at https://github.com/avik-pal/FastStyleTransfer.jl/blob/master/LICENSE.md

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