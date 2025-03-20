# 🛍️ Fashion Recommendation System  

A web-based AI-powered recommendation system that suggests fashion products based on uploaded images. Built with **Streamlit**, **TensorFlow (ResNet50)**, and **Scikit-Learn (Nearest Neighbors)**, it helps users discover similar fashion items instantly.

## 🚀 Features  

- Upload an image to find visually similar fashion products  
- Click on a recommended product to get a fresh set of recommendations  
- Uses **ResNet50** for feature extraction and **Nearest Neighbors** for similarity search  
- Interactive **Streamlit UI** with sidebar options for easy navigation  

## 📸 Screenshots  

### Upload Image  
![Upload Image](images/upload-image.png)  

### Recommendations  
![Recommendations](images/recommendations.png)  

_(Make sure to replace these filenames with actual screenshot paths from your project)_

## 🛠️ Installation & Setup  

Follow these steps to run the project on your local machine:  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/fashion-recommendation.git
cd fashion-recommendation

2️⃣ Install Dependencies
Ensure you have Python installed (3.8+ recommended). Then, install required libraries:


pip install -r requirements.txt
3️⃣ Run the Application

streamlit run app.py
The app will open in your web browser at http://localhost:8501.

📂 Project Structure

📂 fashion-recommendation
 ┣ 📂 images              # Fashion images dataset & screenshots
 ┣ 📂 upload              # Directory for uploaded images
 ┣ 📜 app.py              # Main Streamlit app
 ┣ 📜 Fashion_model.ipynb # Jupyter Notebook for model training
 ┣ 📜 filenames.pkl       # Stored image filenames
 ┣ 📜 Images_features.pkl # Extracted image features
 ┣ 📜 styles.csv          # Metadata for fashion items
 ┣ 📜 README.md           # Project documentation (this file)
🎯 How It Works
Feature Extraction: The app uses ResNet50 to extract visual features from uploaded images.
Similarity Matching: It finds the closest matching products using Nearest Neighbors (Euclidean distance).
Interactive UI: Users can select a recommended product to get continuous recommendations.
🌐 View Live
🔗 Live Demo (Add your deployed app link here)

👨‍💻 Credits
Developed by: [Your Name]
Tech Stack: Python, Streamlit, TensorFlow, Scikit-Learn
Dataset: [Fashion Product Dataset]
