from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import uvicorn
import io
import tensorflow as tf
import PIL.Image

app = FastAPI()

# Load model
MODEL_PATH = r"C:\Users\San_D\Desktop\Project API\CRC\resent50_base_model_wt.h5"
model = load_model(MODEL_PATH)

print("✅ Model Loaded Successfully!")

def preprocess_image(img):
    img = img.convert("RGB")  # Convert TIFF to RGB if needed
    img = img.resize((224, 224))  # ResNet50 input size
    img = np.array(img)  # Convert to NumPy array
    if img.shape != (224, 224, 3):  # Ensure correct shape
        return None
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize
    return img

def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = PIL.Image.open(io.BytesIO(contents)).convert("RGB")  # Ensure compatibility with TIFF
        img_array = preprocess_image(img)
        
        if img_array is None:
            return {"error": "Invalid image shape. Ensure it's a valid 224x224 RGB image."}
        
        prediction = model.predict(img_array)
        
        # Debugging print
        print("Raw Model Prediction:", prediction)
        
        if len(prediction) == 0:
            return {"error": "Model returned an empty prediction list."}
        
        if not isinstance(prediction, np.ndarray):
            return {"error": "Model did not return a NumPy array."}
        
        # Map prediction to class
        class_names = ["Non-Cancerous", "Cancerous"]  # Adjust based on your mapping
        predicted_index = np.argmax(prediction)

        if predicted_index >= len(class_names):  # Ensure it's within bounds
            return {"error": f"Predicted index {predicted_index} is out of range."}

        predicted_class = class_names[predicted_index]

        return {"prediction": predicted_class}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, reload=True)
