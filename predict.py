import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0  
    return np.expand_dims(image, axis=0)  

def predict(image_path, model, top_k):
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)[0]
    
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]  
    top_k_probs = predictions[top_k_indices]
    
    return top_k_indices, top_k_probs


def load_class_names(category_names_path):
    with open(category_names_path, 'r') as f:
        class_names = json.load(f)
    return class_names

def main():
    parser = argparse.ArgumentParser(description="Predict flower class from an image.")
    parser.add_argument('image_path', type=str, help="Path to the image")
    parser.add_argument('model_path', type=str, help="Path to the saved model")
    parser.add_argument('--top_k', type=int, default=5, help="Return top K most likely classes")
    parser.add_argument('--category_names', type=str, help="Path to JSON file mapping labels to flower names")

    args = parser.parse_args()

    
    model = load_model(args.model_path)

   
    top_k_indices, top_k_probs = predict(args.image_path, model, args.top_k)

    
    if args.category_names:
        class_names = load_class_names(args.category_names)
        top_k_labels = [class_names[str(index)] for index in top_k_indices]
    else:
        top_k_labels = top_k_indices  

   
    for i in range(len(top_k_labels)):
        print(f"Class: {top_k_labels[i]}, Probability: {top_k_probs[i]:.4f}")


if __name__ == '__main__':
    main()

   