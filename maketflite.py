import tensorflow as tf
import tensorflow.contrib.lite as lite

converter = lite.TFLiteConverter.from_keras_model_file("sin_model.h5")
tflite_model = converter.convert()
open("sin_model.tflite", "wb").write(tflite_model)
print ("ok")
