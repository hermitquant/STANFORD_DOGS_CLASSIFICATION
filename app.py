import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import os

# Page configuration
st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐕",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 30px;
        background: #e0e0e0;
        border-radius: 15px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        transition: width 0.5s ease;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🐕 Dog Breed Classifier</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<h3>🎯 About This App</h3>
<p>Upload a dog image and get instant breed prediction using our ResNet50 model trained on the Stanford Dogs Dataset.</p>
<p><strong>Model Performance:</strong> 69.44% accuracy on 120 dog breeds</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with model information
st.sidebar.header("📊 Model Information")
st.sidebar.info("""
**Architecture:** ResNet50 (Transfer Learning)
**Training Dataset:** Stanford Dogs Dataset
**Number of Breeds:** 120
**Best Accuracy:** 69.44%
**Input Size:** 224×224 pixels
**Pretrained:** ImageNet weights
""")

# Load breed mapping (you'll need to create this file)
@st.cache_resource
def load_breed_mapping():
    """Load breed to index mapping"""
    # This is a simplified version - you should save your actual breed mapping during training
    breed_list = [
        'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu',
        'Blenheim_spaniel', 'Papillon', 'Toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound',
        'Basset', 'Beagle', 'Bloodhound', 'Bluetick', 'black-and-tan_coonhound',
        'Walker_hound', 'English_foxhound', 'Redbone', 'borzoi', 'Irish_wolfhound',
        'Italian_greyhound', 'Whippet', 'Ibizan_hound', 'Norwegian_elkhound', 'otterhound',
        'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier', 'American_Staffordshire_terrier',
        'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
        'Norwich_terrier', 'Yorkshire_terrier', 'Wire-haired_fox_terrier', 'Lakeland_terrier', 'Sealyham_terrier',
        'Airedale', 'Cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull',
        'Miniature_schnauzer', 'Giant_schnauzer', 'Standard_schnauzer', 'Scotch_terrier', 'Tibetan_terrier',
        'Silky_terrier', 'soft-coated_wheaten_terrier', 'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever',
        'curly-coated_retriever', 'Golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever', 'German_short-haired_pointer',
        'Vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter', 'Brittany_dog',
        'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel', 'Sussex_spaniel',
        'Irish_water_spaniel', 'Kuvasz', 'schipperke', 'groenendael', 'malinois',
        'briard', 'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog',
        'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd',
        'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller',
        'EntleBucher', 'boxer', 'bull_mastiff', 'Tibetan_mastiff', 'French_bulldog',
        'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky',
        'dalmatian', 'dhole', 'dingo', 'foxhound', 'Icelandic_sheepdog',
        'Gordon_setter', 'basenji', 'pug', 'Leonberg', 'Newfoundland',
        'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond',
        'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle',
        'standard_poodle', 'Mexican_hairless', 'timber_wolf', 'white_wolf', 'red_wolf'
    ]
    
    idx_to_breed = {i: f"n02085602-{breed.lower().replace(' ', '_').replace('-', '_')}" for i, breed in enumerate(breed_list)}
    breed_to_idx = {v: k for k, v in idx_to_breed.items()}
    
    return idx_to_breed, breed_to_idx

# Load model
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Check if model file exists
        model_path = "dog_breed_classifier_pretrained.pth"
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please make sure 'dog_breed_classifier_pretrained.pth' is in the same directory as app.py")
            return None
        
        # Load model architecture (without pretrained weights to avoid download issues)
        model = models.resnet50(weights=None)  # Skip download, use your trained weights
        
        # Freeze base layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Replace classifier
        model.fc = nn.Sequential(
            nn.Linear(2048, 512),  # ResNet50 has 2048 features, not 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 120)
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        return model
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

# Make prediction
def predict_breed(image, model, idx_to_breed):
    """Make prediction on uploaded image"""
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get top 5 predictions
        top_prob, top_class = torch.topk(probabilities, 5)
        
        # Convert to readable format
        predictions = []
        for i in range(5):
            breed_name = idx_to_breed[top_class[0][i].item()].split('-')[-1].replace('_', ' ').title()
            confidence_score = top_prob[0][i].item() * 100
            predictions.append({
                'breed': breed_name,
                'confidence': confidence_score,
                'rank': i + 1
            })
        
        return predictions, confidence.item() * 100
    
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        return None, 0

# Main app logic
def main():
    # Load model and breed mapping
    idx_to_breed, breed_to_idx = load_breed_mapping()
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File upload
    st.header("📤 Upload Dog Image")
    
    # Create upload area
    uploaded_file = st.file_uploader(
        "Choose a dog image...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload a clear image of a dog for breed classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        # Two columns layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_column_width=True, caption="Uploaded dog image")
        
        # Make prediction button
        if st.button("🔮 Predict Breed", type="primary"):
            with st.spinner("🔄 Analyzing image..."):
                predictions, top_confidence = predict_breed(image, model, idx_to_breed)
                
                if predictions:
                    # Display results
                    with col2:
                        st.subheader("🎯 Prediction Results")
                        
                        # Top prediction
                        top_pred = predictions[0]
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>🏆 {top_pred['breed']}</h2>
                            <p>Confidence: {top_pred['confidence']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.markdown(f"""
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: {top_pred['confidence']}%"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Top 5 predictions
                        st.subheader("📊 Top 5 Predictions")
                        for pred in predictions:
                            emoji = "🥇" if pred['rank'] == 1 else "🥈" if pred['rank'] == 2 else "🥉" if pred['rank'] == 3 else "🏅"
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem; border-bottom: 1px solid #eee;">
                                <span>{emoji} {pred['breed']}</span>
                                <span><strong>{pred['confidence']:.1f}%</strong></span>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Additional information
                    st.markdown("---")
                    st.subheader("ℹ️ Additional Information")
                    
                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        st.metric("Top Confidence", f"{top_confidence:.1f}%")
                    
                    with col_info2:
                        if top_confidence > 80:
                            st.success("High Confidence")
                        elif top_confidence > 60:
                            st.warning("Medium Confidence")
                        else:
                            st.error("Low Confidence")
                    
                    with col_info3:
                        st.metric("Total Breeds", "120")
                    
                    # Disclaimer
                    st.markdown("""
                    <div class="info-box">
                        <h4>📌 Disclaimer</h4>
                        <p>This AI model is trained on 120 dog breeds with ~69% accuracy. Results may vary based on image quality, angle, lighting, and dog pose. For accurate breed identification, please consult with a veterinarian or breed expert.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    else:
        # Instructions when no image is uploaded
        st.markdown("""
        <div class="upload-area">
            <h3>📸 No Image Uploaded Yet</h3>
            <p>Upload a dog image above to get started with breed classification!</p>
            <p><strong>Tips for best results:</strong></p>
            <ul>
                <li>Use clear, well-lit photos</li>
                <li>Show the dog's face clearly</li>
                <li>Avoid multiple dogs in one image</li>
                <li>Use high-resolution images when possible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
