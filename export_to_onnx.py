import tensorflow as tf
import tf2onnx
import model  # your model.py

# Build and load weights
G2 = model.Generator()
G2.load_weights("weights/generator_epoch_1000.weights.h5")

# Optional: Run once to build the model (especially if using Functional API or Subclassed model)
G2.predict(tf.random.normal([1, 512, 512]))  # Adjust input shape as needed

# Convert to ONNX
spec = (tf.TensorSpec((1, 512, 512), tf.float32),)  # shape should match model input
output_path = "generator.onnx"

model_proto, _ = tf2onnx.convert.from_keras(G2, input_signature=spec, opset=13, output_path=output_path)
print(f"ONNX model saved to {output_path}")
