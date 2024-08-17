# **Context-Aware Image Retrieval System Using Knowledge Graphs and Deep Learning**
## **Overview**
This project develops a Context-Aware Image Retrieval System utilizing Knowledge Graphs and Deep Learning to enhance image retrieval by leveraging contextual understanding.

## **Research Gap**
* Dataset Diversity: Limited multimodal datasets combining images with text, audio, or video.
* Hybrid Models: Scarcity of hybrid models that merge diverse retrieval techniques for improved performance.
* Semantic Gap: Challenges in addressing the semantic gap between images and their associated content, with few existing solutions.
## **Objectives**
* Dataset Creation: Curate a diverse dataset with images and captions.
* Hybrid Model Development: Develop a model that integrates image captioning with Named Entity Recognition (NER).
* Semantic Enhancement: Bridge the semantic gap by combining image captioning, NER, and knowledge graphs.
## **Problem Statement**
Build an innovative image retrieval system integrating image captioning, NER, and knowledge graphs to improve semantic understanding and retrieval accuracy.

## **Workflow**
* Image Feature Extraction: Images are processed through a CNN to extract feature vectors that represent their content at a high level of abstraction.
* Knowledge Graph Integration: The extracted features are mapped to a knowledge graph containing entities and relationships. This integration allows the system to understand the broader context, such as object co-occurrences, spatial relationships, and semantic associations.
* Context-Aware Search: When a query is made, the system not only retrieves images based on feature similarity but also leverages the knowledge graph to infer relevant images based on contextual clues, improving search relevance.
* Retrieval Optimization: By combining deep learning with graph-based reasoning, the system refines search results, prioritizing images that are contextually aligned with the query, offering a more intelligent and accurate retrieval experience.

## **Folder: 8k**
The 8k folder contains all the essential components required for the Context-Aware Image Retrieval System. This includes:

* finalcode.ipynb: The core script implementing Named Entity Recognition (NER) and the Knowledge Graph.
* my_model.h5: The trained model obtained after image captioning.
* graph_data.ttl: The data file for building the knowledge graph structure.
* knowledge_base_data.json: Contains key knowledge base information for integration with the graph.
* indices_to_words.pickle & words_to_indices.pickle: These files map between indices and words for processing textual data.
* output_combinations.csv: Stores output combinations generated after performing NER.

## **Folder: browser**
The browser folder contains all the essential files and directories needed to run the Context-Aware Image Retrieval System. Key components include:

* app.py: The final script that runs the entire system. It initializes the web interface, manages image search requests, and integrates image captioning, NER, and Knowledge Graphs to deliver accurate search results.
* .vscode & pycache: Configuration and cache files supporting the development environment and application performance.
* static, templates, and temp directories: These directories handle the frontend of the web interface, including static assets and templates that render search results.
* captions.txt: A supporting file used for image captioning within the retrieval system.

[Screencast from 15-04-24 09:49:30 PM IST.webm](https://github.com/IndujaSuresh/image-retrieval/assets/69521739/177fd9f2-e964-456e-ae1e-0e44b6c17f6c)
