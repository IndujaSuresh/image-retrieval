#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import json
print("Hi...")
# Define the filename where the knowledge base data is stored
input_filename = "/home/indujasuresh2001/8k/knowledge_base_data (1).json"

# Load the data from the JSON file
with open(input_filename, "r") as json_file:
    loaded_data = json.load(json_file)

# Access the loaded data
knowledge_base_texts = loaded_data["knowledge_base_texts"]
image_names = loaded_data["image_names"]

# Now you can use the loaded data as needed
# For example, printing the first few elements
print("Knowledge Base Texts:")
print(knowledge_base_texts[:5])  # Print the first 5 texts
print("\nImage Names:")
print(image_names[:5])  # Print the first 5 image names


# /kaggle/input/flickr8k/Images/1057089366_ca83da0877.jpg"

# In[2]:


import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import matplotlib.pyplot as plt

# Load the ResNet50 model
resnet_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))

# Load the pre-trained captioning model
caption_model = load_model('/home/indujasuresh2001/8k/my_model.h5')

# Function to preprocess the image and extract ResNet50 features
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to generate captions using the loaded models
def generate_captions(img_path, resnet_model, caption_model, word_to_index, index_to_word):
    in_text = '<start>'
    max_length = 40
    photo = preprocess_image(img_path)
    
    # Extract features using ResNet50
    features = resnet_model.predict(photo)
    
    for _ in range(max_length):
        sequence = [word_to_index[word] for word in in_text.split() if word in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        y_pred = caption_model.predict([features, sequence], verbose=0)
        y_pred = np.argmax(y_pred[0])
        word = index_to_word[y_pred]
        in_text += ' ' + word
        if word == '<end>':
            break
    final = in_text.split()[1:-1]
    sentence = ' '.join(final)
    return sentence

# Example usage: Provide the path to your image
#image_path = "/kaggle/input/flickr8k/Images/1057089366_ca83da0877.jpg"
image_path = "/home/indujasuresh2001/8k/Screenshot from 2024-02-19 20-00-55.png"
# Display the image
img = plt.imread(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
# Load word-to-index and index-to-word mappings
with open("/home/indujasuresh2001/8k/words_to_indices.pickle", "rb") as file:
    word_to_index = pickle.load(file)
with open("/home/indujasuresh2001/8k/indices_to_words.pickle", "rb") as file:
    index_to_word = pickle.load(file)

# Generate captions for the image
generated_caption = generate_captions(image_path, resnet_model, caption_model, word_to_index, index_to_word)
print("Generated Caption:", generated_caption)


# In[54]:



import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance

# Preprocess knowledge_base_texts
processed_texts = [text.split(':')[-1].replace('_', ' ') for text in knowledge_base_texts]

# Calculate Jenson-Shannon divergence
def calculate_jenson_shannon_similarity(input_text, knowledge_base_texts):
    vectorizer = TfidfVectorizer()
    
    # Fit and transform the input text and knowledge_base_texts
    tfidf_matrix = vectorizer.fit_transform([input_text] + knowledge_base_texts)
    
    # Convert the TF-IDF matrix to probability distributions
    input_distribution = tfidf_matrix[0].toarray().flatten()
    knowledge_base_distributions = tfidf_matrix[1:].toarray()

    # Smooth the distributions
    input_distribution = (input_distribution + 1e-10) / np.sum(input_distribution + 1e-10)
    knowledge_base_distributions = (knowledge_base_distributions + 1e-10) / np.sum(knowledge_base_distributions + 1e-10, axis=1, keepdims=True)

    # Calculate Jenson-Shannon divergence
    js_similarities = np.array([1 - distance.jensenshannon(input_distribution, dist) for dist in knowledge_base_distributions])

    return js_similarities

# Input text for comparison
input_text = generated_caption

# Calculate Jenson-Shannon similarity using processed_texts
js_similarity_scores = calculate_jenson_shannon_similarity(input_text, processed_texts)
print("Jenson-Shannon Similarity:", js_similarity_scores)


# In[55]:



import numpy as np

# Assuming you have image_names and js_similarity_scores defined earlier

# Use numpy to get top similar indices
top_similar_indices = np.argsort(js_similarity_scores)[::-1]

# Keep track of selected images
selected_images = set()

# Retrieve top unique similar image names
top_unique_similar_images = []
for idx in top_similar_indices:
    if idx < len(image_names):
        img = image_names[idx]
        similarity_score = js_similarity_scores[idx]

        # Check if the image is not already selected
        if img not in selected_images:
            top_unique_similar_images.append((img, similarity_score))
            selected_images.add(img)

        # Break the loop if we have collected 5 unique images
        if len(top_unique_similar_images) == 5:
            break

# Display top unique similar image names with similarity scores
print("Top unique similar image names with similarity scores:")
for img, similarity_score in top_unique_similar_images:
    print(f"{img} - Similarity Score: {similarity_score}")

# Create a list of image names with the format 'name.jpg'
filtered_top_images = [name[0].split(':')[-1]  for name in top_unique_similar_images]
# Save top unique similar images to a JSON file


import matplotlib.pyplot as plt
from PIL import Image

# Assuming you have a directory path where your images are stored
images_directory = '/home/indujasuresh2001/.kaggle/Images/'  # Replace with your directory path
# Load and display top similar images
for img_file in filtered_top_images:
    img_path = images_directory + img_file
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(img_file)
    plt.axis('off')  # Hide axes
    plt.show()




