# Categorical Modularity: A Tool For Evaluating Word Embeddings

Categorical modularity is a low-resource intrinsic metric for evaluation of word embeddings. We provide code for the community to:
- Generate embedding vectors and nearest-neighbor matrices of lists of core words
- Calculate the following scores for said lists of words:
  - General categorical modularity with respect to a fixed list of semantic categories
  - Single-category modularity with respect to a fixed list of semantic categories
  - Network modularity of emerging categories from the nearest-neighbor graph created by a community detection algorithm
  
 ## Dependencies
 
 ## Calculating Modularity
Given a list of words and a `.bin` or `.vec` embedding model, you can calculate modularities of 
