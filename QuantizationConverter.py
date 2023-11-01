#  -*- coding: utf-8 -*-
#  Author : Subin Lee
#  e-mail : subin.lee@seculayer.com
#  Powered by Seculayer © 2021 Service Model Team, R&D Center.
"""
quantization
"""
import time

import numpy as np
import tensorflow as tf


class QuantizationConverter:
    """quantization and inference"""

    def quantize_and_convert(self):
        """quantize and convert"""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def inference(self, quantized_model_path, x_test, y_test):
        """inference"""
        interpreter = self.load_tflite_model(quantized_model_path)
        input_details, input_index = self.get_model_details(interpreter)

        sum_correct, sum_time = 0.0, 0.0
        idx = -1
        for idx, (image, label) in enumerate(zip(x_test, y_test)):
            image = self.preprocess_input(image, input_details)
            mean_acc, mean_time = self.run_inference(
                interpreter, input_index, image, label
            )
            sum_correct += mean_acc
            sum_time += mean_time

        self.display_results(sum_correct, sum_time, idx)

    def load_tflite_model(self, quantized_model_path):
        """load tflite model"""
        interpreter = tf.lite.Interpreter(model_path=str(quantized_model_path))
        interpreter.allocate_tensors()
        return interpreter

    def get_model_details(self, interpreter):
        """get model details"""
        input_details = interpreter.get_input_details()[0]
        input_index = input_details["index"]
        return input_details, input_index

    def preprocess_input(self, image, input_details):
        """preprocess input"""
        if input_details["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            image = image / input_scale + input_zero_point
            image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
        else:
            image = tf.expand_dims(image, axis=0)
        return image

    def run_inference(self, interpreter, input_index, image, label):
        """run inference"""
        output_index = interpreter.get_output_details()[0]["index"]

        s_time = time.time()
        interpreter.set_tensor(input_index, image)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_index)
        mean_time = time.time() - s_time
        mean_acc = self.calculate_accuracy(pred, label)
        return mean_acc, mean_time

    def calculate_accuracy(self, pred, label):
        """calculate accuracy"""
        return 1.0 if np.argmax(pred) == np.argmax(label) else 0.0

    def display_results(self, sum_correct, sum_time, idx):
        """display results"""
        mean_acc = sum_correct / float(idx + 1)
        mean_time = sum_time / float(idx + 1)
        print(f"Accuracy of quantized model: {mean_acc}")
        print(f"Inference time of the quantized model: {mean_time}")
