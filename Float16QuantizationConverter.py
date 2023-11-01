#  -*- coding: utf-8 -*-
#  Author : Subin Lee
#  e-mail : subin.lee@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.
"""
Quantization
"""
import pathlib

import tensorflow as tf

from load_dataset import load_dataset
from QuantizationConverter import QuantizationConverter


class Float16QuantizationConverter(QuantizationConverter):
    """Float16 Quantization"""

    def quantize_and_convert(self, model_path=None):
        # model load
        keras_model = tf.keras.models.load_model(model_path)

        # Convert to TF Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

        # Convert to TF Lite with float16 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        quantized_tflite_model = converter.convert()

        return quantized_tflite_model


if __name__ == "__main__":
    MODEL_PATH = "./resnet50.h5"
    tflite_models_dir = pathlib.Path("./quantized_model/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    x_train, y_train, x_test, y_test = load_dataset()

    float16_converter = Float16QuantizationConverter()
    float16_quantized_tflite_model = float16_converter.quantize_and_convert(MODEL_PATH)
    float16_qtmodel_path = tflite_models_dir / "float16_quantized.tflite"
    print(
        "float16_quantized model bytes : ",
        float16_qtmodel_path.write_bytes(float16_quantized_tflite_model),
    )
    float16_converter.inference(float16_qtmodel_path, x_test, y_test)
