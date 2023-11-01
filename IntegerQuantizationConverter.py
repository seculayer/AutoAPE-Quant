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


class IntegerQuantizationConverter(QuantizationConverter):
    """Integer Quantization"""

    def quantize_and_convert(self, model_path=None, data=None):
        # define a representative dataset
        def representative_data_gen():
            for input_value in (
                tf.data.Dataset.from_tensor_slices(data).batch(1).take(100)
            ):
                yield [input_value]

        # model load
        keras_model = tf.keras.models.load_model(model_path)

        # Convert to TF Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

        # Convert to TF Lite with integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        quantized_tflite_model = converter.convert()

        return quantized_tflite_model


if __name__ == "__main__":
    MODEL_PATH = "./resnet50.h5"
    tflite_models_dir = pathlib.Path("./quantized_model/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    x_train, y_train, x_test, y_test = load_dataset()

    integer_converter = IntegerQuantizationConverter()
    integer_quantized_tflite_model = integer_converter.quantize_and_convert(
        MODEL_PATH, x_train
    )
    integer_qtmodel_path = tflite_models_dir / "integer_quantized.tflite"
    print(
        "integer_quantized model bytes : ",
        integer_qtmodel_path.write_bytes(integer_quantized_tflite_model),
    )
    integer_converter.inference(integer_qtmodel_path, x_test, y_test)
