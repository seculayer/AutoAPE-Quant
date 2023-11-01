#  -*- coding: utf-8 -*-
#  Author : Subin Lee
#  e-mail : subin.lee@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.
"""
Quantization
"""
import pathlib

import tensorflow as tf
import tensorflow_model_optimization as tfmot

from load_dataset import load_dataset
from QuantizationConverter import QuantizationConverter


class QATQuantizationConverter(QuantizationConverter):
    """Quantization Aware Training"""

    def quantize_and_convert(
        self, model_path=None, data=None, target=None, epochs=None
    ):
        # model load
        keras_model = tf.keras.models.load_model(model_path)

        # q_aware stands for for quantization aware.
        quantize_model = tfmot.quantization.keras.quantize_model
        q_aware_model = quantize_model(keras_model)
        # `quantize_model` requires a recompile.
        q_aware_model.compile(
            optimizer="SGD", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        q_aware_model.fit(data, target, epochs=epochs, batch_size=32)

        # Convert to TF Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
        # Convert to TF Lite with quantization-aware-training
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()

        return quantized_tflite_model


if __name__ == "__main__":
    MODEL_PATH = "./resnet50.h5"
    EPOCHS = 10
    tflite_models_dir = pathlib.Path("./quantized_model/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    x_train, y_train, x_test, y_test = load_dataset()

    qat_converter = QATQuantizationConverter()
    qat_quantized_tflite_model = qat_converter.quantize_and_convert(
        MODEL_PATH, x_train, y_train, EPOCHS
    )
    qat_qtmodel_path = tflite_models_dir / "qat_quantized.tflite"
    print(
        "qat_quantized model bytes : ",
        qat_qtmodel_path.write_bytes(qat_quantized_tflite_model),
    )
    qat_converter.inference(qat_qtmodel_path, x_test, y_test)
