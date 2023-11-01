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


class DynamicQuantizationConverter(QuantizationConverter):
    """Dynamic Quantization"""

    def quantize_and_convert(self, model_path=None):
        # model load
        keras_model = tf.keras.models.load_model(model_path)

        # Convert to TF Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

        # Convert to TF Lite with dynamic quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()

        return quantized_tflite_model


if __name__ == "__main__":
    MODEL_PATH = "./resnet50.h5"
    tflite_models_dir = pathlib.Path("./quantized_model/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    x_train, y_train, x_test, y_test = load_dataset()

    dynamic_converter = DynamicQuantizationConverter()
    dynamic_quantized_tflite_model = dynamic_converter.quantize_and_convert(MODEL_PATH)
    dynamic_qtmodel_path = tflite_models_dir / "dynamic_qauntized.tflite"
    print(
        "dynamic_quantized model bytes : ",
        dynamic_qtmodel_path.write_bytes(dynamic_quantized_tflite_model),
    )
    dynamic_converter.inference(dynamic_qtmodel_path, x_test, y_test)
