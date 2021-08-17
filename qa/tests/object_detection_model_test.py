# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import numpy as np

import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

from PIL import Image

from collections import defaultdict

from itertools import product

from helpers.triton_model_config import Model, TFLiteTritonModel

from timeit import default_timer as timer

def print_timing(start, end):
    print('** Elapsed inference time ', round((end - start) * 1000, 1), 'ms (', round((1.0 / (end - start)), 1), 'fps)')

def extract_photo(filename, width, height, scaling=None):
    img = Image.open(filename)
    resized_img = img.resize((width, height), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    return np.expand_dims(resized, axis=0)


def object_detection_net(
    inference_client,
    client_type,
    test_image,
    expected,
    model_config,
    scaling,
):
    image_input = model_config.inputs[0]
    request_input = client_type.InferInput(
        image_input.name, [1, image_input.dims[1], image_input.dims[1], 3], "UINT8"
    )
    request_input.set_data_from_numpy(
        extract_photo(test_image, image_input.dims[1], image_input.dims[1], scaling)
    )

    request_outputs = []
    for output in model_config.outputs:
        request_outputs.append(client_type.InferRequestedOutput(output.name))
    
    start = timer()

    results = inference_client.infer(
        model_config.name,
        (request_input,),
        model_version="1",
        outputs=request_outputs,
    )

    end = timer()
    print_timing(start, end)

    detection_classes = results.as_numpy(model_config.outputs[1].name)
    detection_probs = results.as_numpy(model_config.outputs[2].name)
    num_detections = results.as_numpy(model_config.outputs[3].name)

    detected_objects_count = defaultdict(lambda: 0)
    for i in range(int(num_detections[0])):
        if detection_probs[0][i] > 0.5:
            detection_class_idx = detection_classes[0][i]
            detected_objects_count[detection_class_idx] += 1

    for class_count in expected:
        assert (
            class_count["count"]
            == detected_objects_count[class_count["detection_index"]]
        )


@pytest.mark.parametrize(
    "model_config",
    [
        TFLiteTritonModel(
            "ssd_mobilenet_v1_coco",
            [
                Model.TensorIO(
                    "normalized_input_image_tensor", "TYPE_UINT8", [1, 300, 300, 3]
                )
            ],
            [
                Model.TensorIO(
                    "TFLite_Detection_PostProcess",
                    "TYPE_FP32",
                    [1, 10, 4],
                ),
                Model.TensorIO(
                    "TFLite_Detection_PostProcess:1",
                    "TYPE_FP32",
                    [1, 10],
                ),
                Model.TensorIO(
                    "TFLite_Detection_PostProcess:2",
                    "TYPE_FP32",
                    [1, 10],
                ),
                Model.TensorIO(
                    "TFLite_Detection_PostProcess:3",
                    "TYPE_FP32",
                    [1],
                ),
            ],
            armnn_cpu=armnn_on,
            xnnpack=False,
        )
        for armnn_on in [False, True]
    ],
)
@pytest.mark.parametrize("client_type", [httpclient])
@pytest.mark.parametrize(
    "test_image,expected",
    [
        ("images/dog.jpg", [{"detection_index": 17, "count": 1}]),
        ("images/people.jpg", [{"detection_index": 0, "count": 3}]),
        ("images/mug.jpg", [{"detection_index": 46, "count": 1}]),
    ],
)
def test_ssd_mobilenet_v1(
    generate_model_config,
    inference_client,
    client_type,
    test_image,
    expected,
    model_config,
):
    start = timer()
    print()
    print(f"test_image = {test_image}, armnn_cpu = {model_config.armnn_cpu}")
    object_detection_net(
        inference_client,
        client_type,
        test_image,
        expected,
        model_config,
        "ssdmobilenetv1",
    )
    end = timer()
    print_timing(start, end)
