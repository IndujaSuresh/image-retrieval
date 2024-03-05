from flask import Flask, render_template, send_from_directory, redirect, url_for
import os
import subprocess
from flask import Flask, render_template, request
from flask import Flask, render_template
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
import os  # Add this import
import pickle
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/submit', methods=['POST'])
def submit():
    # Load data from JSON file
    input_filename = "/image-retrieval/8k/knowledge_base_data (1).json"
    with open(input_filename, "r") as json_file:
        loaded_data = json.load(json_file)
    
    knowledge_base_texts = loaded_data["knowledge_base_texts"]
    image_names = loaded_data["image_names"]

    # Load word-to-index and index-to-word mappings
    with open("/image-retrieval/8k/words_to_indices.pickle", "rb") as file:
        word_to_index = pickle.load(file)
    with open("/image-retrieval/8k/indices_to_words.pickle", "rb") as file:
        index_to_word = pickle.load(file)
    
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

    # Load the ResNet50 model
    resnet_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224, 224, 3))

    # Load the pre-trained captioning model
    caption_model = load_model('/image-retrieval/8k/my_model.h5')

    # Function to preprocess the image and extract ResNet50 features
    def preprocess_image(img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    # Function to generate captions for the image
    #image_path = "/home/indujasuresh2001/8k/Screenshot from 2024-02-19 20-00-55.png"
    if 'uploadedFile' not in request.files:
        return "No file part"
    
    uploaded_file = request.files['uploadedFile']
    print(uploaded_file)  # Print the uploaded file object

    if uploaded_file.filename == '':
        return "No selected file"
    # Now you have the path to the uploaded file
    temp_directory='/home/indujasuresh2001/Browser/temp/'
    uploaded_file.save(os.path.join(temp_directory, uploaded_file.filename))

   
    image_path = os.path.join(temp_directory, uploaded_file.filename)

    generated_caption = generate_captions(image_path, resnet_model, caption_model, word_to_index, index_to_word)
    print(generated_caption)
    # Calculate Jenson-Shannon similarity using processed_texts
    js_similarity_scores = calculate_jenson_shannon_similarity(generated_caption, processed_texts)
    
    # Assuming you have image_names and js_similarity_scores defined earlier
    top_similar_indices = np.argsort(js_similarity_scores)[::-1]
    selected_images = set()
    top_unique_similar_images = []
    for idx in top_similar_indices:
        if idx < len(image_names):
            img = image_names[idx]
            similarity_score = js_similarity_scores[idx]

            if img not in selected_images:
                top_unique_similar_images.append((img, similarity_score))
                selected_images.add(img)

            if len(top_unique_similar_images) == 5:
                break

    filtered_top_images = [name[0].split(':')[-1]  for name in top_unique_similar_images]
    print(filtered_top_images )
    # Create a list of image paths
    images_directory = 'static/Images/'  # Replace with your directory path
    image_paths = [os.path.join(images_directory, img) for img in filtered_top_images]
    
    return render_template('nextpage.html', images=image_paths)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    #server_port = os.environ.get('PORT', '8080')
    #app.run(debug=False, port=server_port, host='0.0.0.0')
    app.run()

