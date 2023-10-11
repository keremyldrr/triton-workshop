#!/usr/bin/env python3

import ast
import urllib
from io import BytesIO

import numpy as np
import requests
import tensorflow as tf
import tritonclient.grpc as grpc_client
from PIL import Image

# Get imagenet class names as dictionary
r = urllib.request.urlopen(
    "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
).read()
class2name = ast.literal_eval(r.decode())


def get_input_output_names_shapes(
    client: grpc_client.InferenceServerClient, model_name: str
):
    model_metadata = client.get_model_metadata(model_name)
    input_names = [input_tensor.name for input_tensor in model_metadata.inputs]
    output_names = [output_tensor.name for output_tensor in model_metadata.outputs]

    model_config = client.get_model_config(model_name)
    input_shapes = [input_tensor.dims for input_tensor in model_config.config.input]

    return input_names, output_names, input_shapes


def main():
    # Url to retrieve an image from the internet
    url = "https://i1.wp.com/robinbarefield.com/wp-content/uploads/2015/03/DSC_1763.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))  # read the image with PIL

    model_name = "resnet50"  # Set the model name
    with grpc_client.InferenceServerClient(
        "localhost:8001"
    ) as client:  # connect to the server
        # Get the names and shapes of the input and output tensors
        input_names, output_names, input_shapes = get_input_output_names_shapes(
            client, model_name
        )

        # Resize the image to expected input shape of the model
        width, height = input_shapes[0][0], input_shapes[0][1]
        img = img.resize((width, height))

        # Convert the downloaded image to RGB
        input_data = np.array(img.convert("RGB")).astype(np.float32)

        # Preprocess the image for the ResNet50 model
        input_data = tf.keras.applications.resnet50.preprocess_input(input_data)

        # Add the dimension for the batch size
        input_data = np.expand_dims(input_data, 0)

        # Create and fill the actual input tensor to the model
        input_tensor = grpc_client.InferInput(input_names[0], input_data.shape, "FP32")
        input_tensor.set_data_from_numpy(input_data)

        # Run inference
        inputs = [input_tensor]
        outputs = [
            grpc_client.InferRequestedOutput(output_name)
            for output_name in output_names
        ]
        result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

        # Print out predicted class name
        print(class2name[np.argmax(result.as_numpy(output_names[0]))])


if __name__ == "__main__":
    main()
