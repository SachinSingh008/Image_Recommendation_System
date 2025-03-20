import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st
import pandas as pd  

# --------------- Streamlit UI Setup ---------------
st.set_page_config(page_title="Fashion Recommendation System", layout="wide")

# Add custom styling
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #FF5733;
        text-align: center;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #1E90FF;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------- App Header ---------------
st.markdown('<p class="title"><h1> 👗Fashion Recommendation System</h1></p>', unsafe_allow_html=True)
st.write("Click on a recommended product to get new recommendations!")

# --------------- Load Model & Data ---------------
if not os.path.exists('Images_features.pkl') or not os.path.exists('filenames.pkl') or not os.path.exists('styles.csv'):
    st.error("⚠️ Required files (Images_features.pkl, filenames.pkl, or styles.csv) not found!")
    st.stop()

# Load image features & filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Load product details
df = pd.read_csv('styles.csv')

# Convert column names to lowercase for consistency
df.columns = df.columns.str.lower()
df["id"] = df["id"].astype(str)  # Ensure 'id' is treated as a string for matching

# Function to extract features from uploaded images
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)

    with tf.device('/CPU:0'):  # Run on CPU to avoid GPU issues
        result = model.predict(img_preprocess).flatten()

    return result / norm(result)

# Load & freeze ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Fit Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Initialize session state variables
if "selected_index" not in st.session_state:
    st.session_state.selected_index = None  # Stores the currently selected product index
if "current_recommendations" not in st.session_state:
    st.session_state.current_recommendations = []  # Stores recommendations for the selected product
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None  # Store the last uploaded file

# Reset function to allow new uploads
def reset_recommendation():
    st.session_state.selected_index = None
    st.session_state.current_recommendations = []
    st.session_state.uploaded_file = None
    st.rerun()  # Refresh UI

# Sidebar for Uploading Image
st.sidebar.header("Upload Image")
upload_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

# Add Reset Button
if st.sidebar.button("🔄 Reset & Upload New Image"):
    reset_recommendation()

# If a new file is uploaded, reset previous recommendations
if upload_file is not None and upload_file != st.session_state.uploaded_file:
    st.session_state.uploaded_file = upload_file
    os.makedirs("upload", exist_ok=True)
    file_path = os.path.join("upload", upload_file.name)

    with open(file_path, 'wb') as f:
        f.write(upload_file.getbuffer())

    # Display uploaded image
    st.sidebar.image(file_path, caption="Uploaded Image", width=200)

    # Extract features & get initial recommendations
    with st.spinner("Processing image..."):
        input_img_features = extract_features_from_images(file_path, model)
        distance, indices = neighbors.kneighbors([input_img_features])

    # Set the first recommended product as the selected index
    st.session_state.selected_index = indices[0][1]  # Select the best match
    st.session_state.current_recommendations = indices[0][2:]  # Store other recommendations

# If a product is selected, display its details
if st.session_state.selected_index is not None:
    selected_index = st.session_state.selected_index
    selected_filename = filenames[selected_index]
    product_id = os.path.basename(selected_filename).split(".")[0]  # Extract product ID

    # Get product details
    product_info = df[df["id"] == product_id]

    if not product_info.empty:
        selected_product = product_info.iloc[0]  # Get first matching row
        st.markdown('<p class="subheader">📌 Product Details</p>', unsafe_allow_html=True)
        st.image(selected_filename, width=300)

        # Show product details
        st.write(f"**Category:** {selected_product['mastercategory']}")
        st.write(f"**Subcategory:** {selected_product['subcategory']}")
        st.write(f"**Article Type:** {selected_product['articletype']}")
        st.write(f"**Base Colour:** {selected_product['basecolour']}")
        st.write(f"**Season:** {selected_product['season']}")
        st.write(f"**Year:** {selected_product['year']}")
        st.write(f"**Usage:** {selected_product['usage']}")
        st.write(f"**Product Name:** {selected_product['productdisplayname']}")

        # Display new recommendations
        st.markdown('<p class="subheader">🔄 Recommended Products</p>', unsafe_allow_html=True)

        new_recommendations = st.session_state.current_recommendations
        cols = st.columns(len(new_recommendations))

        def update_recommendations(index):
            """Update session state to show new recommendations."""
            st.session_state.selected_index = index
            st.session_state.current_recommendations = neighbors.kneighbors([Image_features[index]], n_neighbors=6)[1][0][1:]  # Exclude selected item itself
            st.rerun()

        for i, rec_index in enumerate(new_recommendations):
            with cols[i]:
                rec_filename = filenames[rec_index]

                # Clicking on a recommendation updates the session state
                if st.button(f"View {i+1}"):
                    update_recommendations(rec_index)

                st.image(rec_filename, caption=f"Similar {i+1}")

    else:
        st.warning("⚠️ No product details found for this image!")
