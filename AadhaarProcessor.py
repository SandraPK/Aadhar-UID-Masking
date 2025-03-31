import cv2
import torch
import open_clip
import pytesseract
from sentence_transformers import util
from PIL import Image
import torchvision.transforms as T
import os
import time
import re
from rembg import remove
from cv2 import imread
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
import pyzipper
from timeit import default_timer as timer
import base64
import requests
import io
from cryptography.fernet import Fernet
import json
import re
import sys



face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# Function to load the saved model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model
# Function for image classification
def classify_image(image_path, model):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction
        
def save_image_in_protected_zip(image_path, zip_path, password):
    # Create the zip file with password protection using pyzipper
    with pyzipper.AESZipFile(zip_path, 'w', compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zipf:
        zipf.setpassword(password.encode())  # Set password for the ZIP archive
        zipf.write(image_path, os.path.basename(image_path))
    
    print(f"The image has been saved to {zip_path} and password-protected.")
def protect_image(image_path, storage_folder,customer_id):
    # Get the password from the user
    password = customer_id
    # Ensure the storage folder exists
    if not os.path.exists(storage_folder):
        os.makedirs(storage_folder)
    
    # Generate a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    zip_filename = f"protected_image_{timestamp}.zip"
    zip_path = os.path.join(storage_folder, zip_filename)
    
    # Save the image in a password-protected zip file
    save_image_in_protected_zip(image_path, zip_path, password)

def extract_and_mask_aadhaar(image, output_path,corrected_path,reference_id):
    storage_folder = "protected_images_storage"
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Define the expected region for the Aadhaar number
    x_min, y_min, x_max, y_max = 80, 350, 430, 480
    # Extract the region of interest (ROI) where the Aadhaar number is expected
    roi = gray_image[y_min:y_max, x_min:x_max]
    # Using pytesseract to extract text information
    data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
    print(data)
    # Regular expression for Aadhaar number segment
    aadhaar_patterns = [
        re.compile(r'\b\d{4}\b'),
        re.compile(r'\b\d{4} \d{4} \d{4}\b'),
        re.compile(r'\d{8}\b'),
        re.compile(r'\b\d{8} \d{4}\b')
        
    ]
    found = False
    digit_count = 0
    max_digits = 12
    aadhaar_segments = []
    # Loop through the detected text
    for i, text in enumerate(data['text']):
        if any(pattern.fullmatch(text.strip()) for pattern in aadhaar_patterns):
            aadhaar_segment = text.strip()
            aadhaar_segments.append((aadhaar_segment, data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
            digit_count += len(aadhaar_segment)
            # Check if the segment is located in the expected region
            if digit_count >= max_digits:
                print(f"Found Aadhaar number segments: {aadhaar_segments}")
                # Mask the first 8 digits of the Aadhaar number
                chars_to_mask = digit_count - 4
                for segment in aadhaar_segments:
                    seg_text, seg_x, seg_y, seg_w, seg_h = segment
                    if chars_to_mask > 0:
                        mask_length = min(len(seg_text), chars_to_mask)
                        if mask_length > 0:
                            # Mask only the required part of the segment
                            mask_w = int(seg_w * (mask_length / len(seg_text)))
                            cv2.rectangle(image, (x_min + seg_x, y_min + seg_y), (x_min + seg_x + mask_w, y_min + seg_y + seg_h), (0, 0, 0), -1)
                        chars_to_mask -= mask_length
                    else:
                        break
                found = True
                break  # Exit loop after masking the valid Aadhaar number segments
    
    if found:
        # Ensure the output folder exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Generate a unique filename using a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        output_path = os.path.join(output_path, f"masked_image_{timestamp}.png")
        
        # Save the updated image
        cv2.imwrite(output_path, image)
        print(f"Masked Aadhaar number saved to {output_path}")
        is_number=True
        protect_image(corrected_path, storage_folder,reference_id)
    else:
        print("No valid Aadhaar number found in the image.")
        print("check another case")
        x_min, y_min, x_max, y_max = 80, 310, 430, 430
        # Extract the region of interest (ROI) where the Aadhaar number is expected
        roi = gray_image[y_min:y_max, x_min:x_max]
        # Using pytesseract to extract text information
        data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
        print(data)
        # Regular expression for Aadhaar number segment
        aadhaar_patterns = [
            re.compile(r'\b\d{4}\b'),
            re.compile(r'\b\d{4} \d{4} \d{4}\b'),
            re.compile(r'\d{8}\b'),
            re.compile(r'\b\d{8} \d{4}\b'),
            re.compile(r'\d{12}\b')
        ]
        found = False
        aadhaar_segments = []
        # Loop through the detected text
        for i, text in enumerate(data['text']):
            if any(pattern.fullmatch(text.strip()) for pattern in aadhaar_patterns):
                aadhaar_segment = text.strip()
                aadhaar_segments.append((aadhaar_segment, data['left'][i], data['top'][i], data['width'][i], data['height'][i]))
                
                # Check if the full 12-digit Aadhaar number is identified
                combined_aadhaar = ''.join([seg[0] for seg in aadhaar_segments])
                if len(combined_aadhaar) >= 12:
                    combined_aadhaar = combined_aadhaar[:12]
                    print(f"Found Aadhaar number segments: {combined_aadhaar}")
                    # Mask the first 8 digits of the Aadhaar number
                    chars_to_mask = 8
                    for segment in aadhaar_segments:
                        seg_text, seg_x, seg_y, seg_w, seg_h = segment
                        mask_length = min(len(seg_text), chars_to_mask)
                        if mask_length > 0:
                            # Mask only the required part of the segment
                            mask_w = int(seg_w * (mask_length / len(seg_text)))
                            cv2.rectangle(image, (x_min + seg_x, y_min + seg_y), (x_min + seg_x + mask_w, y_min + seg_y + seg_h), (0, 0, 0), -1)
                        chars_to_mask -= mask_length
                        if chars_to_mask <= 0:
                            break
                    # Update segment for the last 4 digits to remain visible
                    for segment in aadhaar_segments:
                        seg_text, seg_x, seg_y, seg_w, seg_h = segment
                        visible_length = len(seg_text) - max(0, chars_to_mask)
                        if visible_length < len(seg_text):
                            mask_w = int(seg_w * (visible_length / len(seg_text)))
                            cv2.rectangle(image, (x_min + seg_x + mask_w, y_min + seg_y), (x_min + seg_x + seg_w, y_min + seg_y + seg_h), (255, 255, 255), -1)
                        chars_to_mask -= len(seg_text)
                    found = True
                    break  # Exit loop after masking the valid Aadhaar number segments
        if found:
            # Ensure the output folder exists
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            # Generate a unique filename using a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            output_path = os.path.join(output_path, f"masked_image_{timestamp}.png")
            
            # Save the updated image
            cv2.imwrite(output_path, image)
            print(f"Masked Aadhaar number saved to {output_path}")
            is_number=True
            protect_image(corrected_path, storage_folder,reference_id)
        else:
            print("No valid Aadhaar number found in the image.")
            is_number=False
            
    return image,is_number
def check_front(resized_image2):
    check=False
    image=resized_image2.copy()
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40))
    
    def imageEncoder(img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        img1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img1 = transform(img1).unsqueeze(0).to(device)
        img1 = model.encode_image(img1)
        return img1
    
    # Image processing model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained="openai")
    model.to(device)
    # Additional preprocessing steps
    transform = T.Compose([
        T.Resize((336, 336)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    def generateScore(image1, image2):
        #test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
        test_img = Image.open(image1)
        #data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
        data_img = Image.open(image2)
        img1 = imageEncoder(test_img)
        img2 = imageEncoder(data_img)
        cos_scores = util.pytorch_cos_sim(img1, img2)
        score = round(float(cos_scores[0][0])*100, 2)
        return score
    similarity_score = generateScore("Screenshot 2024-03-28 110758.png", 'goi_symbol_image.png')
    
    
    if len(faces) > 0 or similarity_score>=80:
        check=True
    else:
        check=False
    return check,similarity_score
def extract_details(image,corrected_path,reference_id):
    
    x_min, y_min, x_max, y_max = 80, 350, 430, 440  # Coordinates to crop Aadhaar number
    # Extract the Aadhaar number region
    no = image[y_min:y_max, x_min:x_max]
    cv2.imwrite('number.png', no)
    
    output_path = "maskedimages"
    image,is_number=extract_and_mask_aadhaar(image, output_path,corrected_path,reference_id)
    
    return image,is_number
def extract_details_back(image,corrected_path,reference_id):
    
    x_min, y_min, x_max, y_max = 80, 350, 430, 440  # Coordinates to crop Aadhaar number
    # Extract the Aadhaar number region
    no = image[y_min:y_max, x_min:x_max]
    cv2.imwrite('number.png', no)
    
    output_path = "maskedimages"
    image,is_number=extract_and_mask_aadhaar(image, output_path,corrected_path,reference_id)
    
    return image,is_number
def resizingimg(image,image_dup):
    newsize = (500, 500)
    resized_image1 = cv2.resize(image,newsize)
    resized_dup=cv2.resize(image_dup,newsize)
    return resized_image1,resized_dup
def rotate_image(image, i, image_dup):
    for j in range(i):
        image,image_dup = resizingimg(image,image_dup)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image_dup = cv2.rotate(image_dup, cv2.ROTATE_90_CLOCKWISE)
        j += 1
    return image, image_dup
def position(resized_image): 
    resized_image_dup = np.copy(resized_image)
    
    score=False 
    def imageEncoder(img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        img1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img1 = transform(img1).unsqueeze(0).to(device)
        img1 = model.encode_image(img1)
        return img1
    
    def find_position(resized_image):
        x_min, y_min, x_max, y_max = 390, 20, 500, 130 
        # Draw a green rectangle around the GOI symbol
        cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        # Extract the GOI symbol image
        aadhar_emblem_image = resized_image[y_min:y_max, x_min:x_max]
        
        return aadhar_emblem_image
        
    def calculate_emblem_similarity(aadhar_emblem_image, emblem_reference_path):
        emblem_reference = Image.open(emblem_reference_path)
        aadhar_emblem_encoded = imageEncoder(aadhar_emblem_image)
        emblem_encoded = imageEncoder(emblem_reference)
        emblem_similarity_score1 = util.pytorch_cos_sim(aadhar_emblem_encoded, emblem_encoded)
        emblem_similarity_score1 = round(float(emblem_similarity_score1[0][0]) * 100, 2)
        return emblem_similarity_score1
    # Apply thresholding
    _, thresh = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY_INV)
    x, y, w, h = 10, 5, 93, 93
    # Draw a rectangle around the red box in emblem
    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # Extract the red box image
    red_box_image = resized_image[y:y+h, x:x+w]
    x_min, y_min, x_max, y_max = 110, 20, 500, 100 
    # Draw a green rectangle around the GOI symbol
    cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    # Extract the GOI symbol image
    goi_symbol_image = resized_image[y_min:y_max, x_min:x_max]
    # Save the extracted images 
    cv2.imwrite('red_box_image.png', red_box_image) 
    cv2.imwrite('goi_symbol_image.png', goi_symbol_image)
    # Image processing model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained="openai")
    model.to(device)
    # Additional preprocessing steps
    transform = T.Compose([
        T.Resize((336, 336)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    emblem_reference_path= "Screenshot 2024-03-28 121120.png"
    emblem_similarity_score=calculate_emblem_similarity(red_box_image, emblem_reference_path)
    if emblem_similarity_score>=80:
        stat=True
        if stat:
            aadhar_emblem_image=find_position(resized_image)
            # Save the extracted images 
            cv2.imwrite('aadhar_emblem_image.png', aadhar_emblem_image)
            emblem_reference_path= "Screenshot 2024-05-22 102016.png"
            emblem_similarity_score1 = calculate_emblem_similarity(aadhar_emblem_image, emblem_reference_path)
            stat=False
            if emblem_similarity_score1>=80 or emblem_similarity_score>=85:
                print(f"The input matches with a combined similarity score of {emblem_similarity_score}%.")
                score=True
                stat=True
        else:
            print(f"The input does not match (combined similarity score: {emblem_similarity_score}%).")
            
    else:
        stat=False
        print("Check another method")
        x, y, w, h = 15, 15, 110, 110
        # Draw a rectangle around the red box in emblem
        cv2.rectangle(resized_image_dup, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # Extract the red box image
        red_box_image = resized_image_dup[y:y+h, x:x+w]
        # Save the extracted images 
        cv2.imwrite('red_box_image.png', red_box_image) 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained="openai")
        model.to(device)
        # Additional preprocessing steps
        transform = T.Compose([
            T.Resize((336, 336)),
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        emblem_reference="Screenshot 2024-11-25 122208.png"
        emblem_similarity_score=calculate_emblem_similarity(red_box_image, emblem_reference)
        if emblem_similarity_score>=80:
            stat=True
            if stat:
                aadhar_emblem_image=find_position(resized_image)
                # Save the extracted images 
                cv2.imwrite('aadhar_emblem_image.png', aadhar_emblem_image)
                emblem_reference_path= "Screenshot 2024-05-22 102016.png"
                emblem_similarity_score1=calculate_emblem_similarity(aadhar_emblem_image, emblem_reference_path)
                stat=False
                if emblem_similarity_score1>=80 or emblem_similarity_score>=85:
                    print(f"The input matches with a combined similarity score of {emblem_similarity_score}%.")
                    score=True
                    stat=True
            else:
                print(f"The input does not match (combined similarity score: {emblem_similarity_score}%).")
                
        else:
            stat=False
            emblem_reference="Screenshot 2024-03-28 121120.png"
            emblem_similarity_score=calculate_emblem_similarity(red_box_image, emblem_reference)
            if emblem_similarity_score>=80:
                print(f"The input matches with a combined similarity score of {emblem_similarity_score}%.")
                score=True
                stat=True
        
    return stat,score
class AadhaarProcessor:
    @staticmethod
    def test(reference_id,img,file_formats):
        try:
            if not img:
                print(f"No image provided for reference ID: {reference_id}")
                return
            if any(img.endswith(file_format) for file_format in file_formats):
                start = timer()
                print(f"Processing image: {img}")
                remote_image_file = f"{img}"
                corrected_path = remote_image_file.replace('\\', '/')
                image = imread(corrected_path)
                # Load the saved model
                model_path = 'background_classifier.h5'
                loaded_model = load_model(model_path)
                prediction = classify_image(corrected_path, loaded_model)
                # Interpret the prediction
                if prediction[0][0] < 0.5:
                    print("Background")
                    surround = True
                else:
                    print("Non-background")
                    surround = False
        except Exception as e:
            print(f"Error occurred while processing {reference_id}: {e}")
        return surround,corrected_path, image,start
        
    @staticmethod
    def crop_image(image_path):
        input_image = Image.open(image_path)
        output_image = remove(input_image)
        output = 'cropped_output.png'
        output_image.save(output)
        
        output_array = np.array(output_image)
        gray_image = cv2.cvtColor(output_array, cv2.COLOR_RGBA2GRAY if output_array.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min, y_min, x_max, y_max = min(x_min, x), min(y_min, y), max(x_max, x + w), max(y_max, y + h)
        
        document_image = output_image.crop((x_min, y_min, x_max, y_max))
        document_image.save('document_image.png')
        return document_image
    
    @staticmethod
    def mask_aadhaar_number(image, output_folder, reference_id):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x_min, y_min, x_max, y_max = 80, 350, 430, 480
        roi = gray_image[y_min:y_max, x_min:x_max]
        
        data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
        aadhaar_pattern = r'\b\d{4} \d{4} \d{4}\b'
        for text in data['text']:
            if re.match(aadhaar_pattern, text):
                # Mask Aadhaar number
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
                break
        
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        output_path = os.path.join(output_folder, f"masked_image_{timestamp}.png")
        cv2.imwrite(output_path, image)
        return output_path
    
    @staticmethod
    def correct_orientation(image):
        for i in range(0, 5):
            path="document_image.png"
            image = cv2.imread(path)
            image_dup=image
            if i == 0:
                resized_image2,resized_dup = resizingimg(image,image_dup)
                stat,score = position(resized_image2)
                if stat==True:
                    break
            elif i == 1:
                resized_image2,resized_dup = rotate_image(image,i, image_dup)
                stat,score= position(resized_image2)
                if stat==True:
                    break
            else:
                if not stat:
                    resized_image2,resized_dup = rotate_image(image,i, image_dup)
                    stat,score= position(resized_image2)
                elif stat and score:
                    resized_image2 = resized_image2
                    resized_dup=resized_dup
                    break
                elif not score:
                    print("Invalid Document")               
            i += 1
        return resized_dup,score
    @staticmethod
    
    def process_image(reference_id, document_type, image_path):
        while True:
            if document_type == 'Aadhar':
                file_formats=['jpg','jpeg']
                pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\101066\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

                surround,corrected_path,image,start=AadhaarProcessor.test(reference_id,image_path,file_formats)
                if surround:
                    image = AadhaarProcessor.crop_image(corrected_path)
                else:
                    cv2.imwrite('document_image.png', image)
                resized_image2, score = AadhaarProcessor.correct_orientation(image)
                if score:
                    check, similarity_score = check_front(resized_image2)
                    if check:
                        image ,is_number= extract_details(resized_image2, corrected_path, reference_id)
                        if is_number:
                            img=image
                            break
                        else:
                            data="Invalid Document"
                            img=None
                            #delete_files_in_directory_and_subdirectories(image_path)
                            break
                    elif similarity_score >= 80:
                        image,is_number = extract_details_back(resized_image2, corrected_path, reference_id)
                        if is_number:
                            
                            data = 'Valid Document'
                            img=image
                            #delete_files_in_directory_and_subdirectories(image_path)
                            break
                        else:
                            data = "Invalid Document"
                            img=None
                            #delete_files_in_directory_and_subdirectories(image_path)
                            break
                    else:
                        data="Invalid Document"
                        img=None
                        break
                else:
                    data = "Invalid Document"
                    img=None
                    break
        return img
         
class ImageProcessor:
    @staticmethod
    def save_base64_image(base64_string, save_path):
        try:
            image_data = base64.b64decode(base64_string)
            with open(save_path, 'wb') as file:
                file.write(image_data)
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    @staticmethod
    def encode_image_to_base64(image_buffer):
        return base64.b64encode(image_buffer.read()).decode('utf-8')

import os
import streamlit as st

class FolderMonitor:
    def __init__(self, image_processor, image_folder):
        self.image_processor = image_processor
        self.image_folder = image_folder
        self.processed_folder = os.path.join(image_folder, "processed")
        os.makedirs(self.processed_folder, exist_ok=True)

    def process_image(self, reference_id, image_path):
        processed_image = AadhaarProcessor.process_image(reference_id, 'Aadhar', image_path)

        # Save processed image
        processed_image_path = os.path.join(self.processed_folder, f"{reference_id}.jpg")
        os.rename(image_path, processed_image_path)
        return processed_image, processed_image_path

def run_monitor(image_processor, image_folder):
    st.title("Local Aadhaar Processing")
    reference_id = st.text_input("Enter Reference ID:")
    uploaded_file = st.file_uploader("Upload Image:", type=["jpg", "jpeg", "png"])

    if st.button("Process"):
        if reference_id and uploaded_file:
            image_path = os.path.join(image_folder, f"{reference_id}.jpg")
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())
            
            monitor = FolderMonitor(image_processor, image_folder)
            processed_image_path = monitor.process_image(reference_id, image_path)
            
            st.image(processed_image_path, caption="Processed Image")
        else:
            st.error("Please provide both a reference ID and an image.")

if __name__ == "__main__":
    image_folder = "images"  # Replace with your local image folder path
    os.makedirs(image_folder, exist_ok=True)
    image_processor = ImageProcessor()  # Assume this is properly defined elsewhere
    run_monitor(image_processor, image_folder)


