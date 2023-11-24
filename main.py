from fastapi import FastAPI, UploadFile, HTTPException
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Allow all origins for CORS
origins = ["*"]

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your Keras model
model = load_model('model_vgg19.h5')

def predict(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)
        classes = model.predict(img_data)
        malignant = float(classes[0, 0])  # Convert to float
        normal = float(classes[0, 1])     # Convert to float

        return malignant, normal
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    finally:
        # Clean up the temporary image file
        if os.path.exists(image_path):
            os.remove(image_path)

@app.post("/predict")
async def predict_image(file: UploadFile):
    temp_image_path = "temp_image.jpg"
    
    try:
        # Save the uploaded image temporarily
        with open(temp_image_path, "wb") as temp_image:
            temp_image.write(file.file.read())

        # Perform prediction on the saved image
        malignant, normal = predict(temp_image_path)

        if malignant > normal:
            prediction = 'malignant'
        else:
            prediction = 'normal'

        # Convert NumPy floats to Python floats
        malignant = float(malignant)
        normal = float(normal)

        return {"prediction": prediction, "malignant_prob": malignant, "normal_prob": normal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")
    finally:
        # Ensure cleanup even if an exception occurs
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
