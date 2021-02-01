# Categorical Modularity: A Tool For Evaluating Word Embeddings

Categorical modularity is a low-resource intrinsic metric for evaluation of word embeddings. We provide code for the community to:
- Generate embedding vectors and nearest-neighbor matrices of lists of core words
- Calculate the following scores for said lists of words:
  - General categorical modularity with respect to a fixed list of semantic categories
  - Single-category modularity with respect to a fixed list of semantic categories
  - Network modularity of emerging categories from the nearest-neighbor graph created by a community detection algorithm
  
 ## Dependencies
 - Python 3.6+
 - scipy
 - numpy
 - nltk
 - fasttext
 - google_trans_new
 
 ## Calculating Modularity
Given a list of words and a `.bin` or `.vec` embedding model, you can calculate several modularity metrics using the files in the `core` directory. Brief descriptions of files and functionalities:
- `core/ftvectorgen.py`: pass in a file containing a list of words and a file containing an embedding model binary and create a file with 300-dimensional embedding vectors of each of the input words. Use for FastText-compatible models (e.g. FastText, subs2vec). Usage (with respect to topmost level directory of this repo): 
```
python3 core/ftvectorgen.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUT_FILE
```
- `core/musevectorgen.py`: same functionality as `core/ftvectorgen.py`, but used for MUSE-compatible models (a different implementation than the FastText library). Additionally asks for a language parameter specification for use in MUSE embedding generation. Usage:
```
python3 core/musevectorgen.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUT_FILE --language LANGUAGE
```
- `core/ftmatrices.py`: given a list of n words and an embedding model binary, generates a file with an n x n matrix where row i, column j = k represents the fact that word j is the kth-nearest neighbor of word i in the given embedding space. Use this file for FastText-compatible models only. Usage:
```
python3 core/ftmatrices.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUTFILE
```
- `core/musematrices.py`: same functionality as `core/ftmatrices.py` but for MUSE-compatible model binaries. Usage: 
```
python3 core/musematrices.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUTFILE --language LANGUAGE
```
- `core/modularity.ipynb`: an interactive Python notebook that contains code that can be used to calculate  general categorical modularity and unsupervised cluster modularity. Each of these is one cell of the notebook. WILL CHANGE THIS ONCE WE SEPARATE TO INDIVIDUAL FILES
- `core/correlationcalc.py`: code that can be used to calculate the Spearman rank correlations of one set of modularity scores with one set of task performance metrics. Usage:
```
python3 core/correlationcalc.py --modularity_file MODULARITY_FILE --downstream_file DOWNSTREAM_FILE
```

## Single-Category Modularity Correlations
Our paper also explores an extension of categorical modularity to single-category modularity, which we test on each of the 59 categories listed in our paper. The `single_category` directory contains code that can be used to calculate these single-category modularities and their correlations with downstream task performance. Brief descriptions of files and functionalities:
- `single_category/singlecatcorrelationcalc.py`: given a file with modularity scores for a set of categories and a file with performance metrics for a particular tasks, writes an output file with correlations between performance metrics and modularities with respect to each category. Usage:
```
python3 single_category/singlecatcorrelationcalc.py --modularity_file MODULARITY_FILE --metrics_file MODULARITY_FILE --out_file OUT_FILE
```
- `single_category/singlecatmodularity.py`: given a list of category labels for words and a square matrix of nearest neighbor relationships among words, calculates single-category modularities for each category. Usage: SILVIA UPDATE THIS


## Running Downstream Tasks
Our paper presents moderate to strong correlations of categorical modularity with four downstream tasks. We provide code to reproduce these tasks in the `task_bli`, `task_wordsim`, and `task_movies` directories.
### Sentiment Analysis
The first task we run is sentiment analysis of [IMDB movie reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Files and functionalities:
- `task_movies/moviedatagen.py`: given a file with data (raw text of movie reviews) and a target language, generates an equivalent dataset translated into the target language. Usage:
```
python3 task_movies/moviedatagen.py --data_file DATA_FILE --target_language_name TARGE_LANGUAGE_NAME --target_language_code TARGET_LANGUAGE_CODE
```
- `task_movies/ftmoviegen.py`: given a data file with raw movie reviews and a model binary file, produces an output file 300-dimensional embeddings of each review. Use only for FastText-compatible models. Usage:
```
python3 task_movies/ftmoviegen.py --data_file DATA_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_movies/musemoviegen.py`: given a data file with raw movie reviews and a model binary file, produces an output file 300-dimensional embeddings of each review. Use only for MUSE-compatible models. Usage:
```
python3 task_movies/musemoviegen.py --data_file DATA_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_movies/movietask.py`: given data in the form of vectors (assuming the first half are positive and the second half are negative), runs the task of sentiment analysis and outputs mean accuracy and precision over 30 trials. Usage:
```
python3 task_movies/movietask.py --data_file DATA_FILE --model_name MODEL_NAME --num_trials NUM_TRIALS --dataset_size DATASET_SIZE --train_proportion TRAIN_PROPORTION --language LANGUAGE
```

### Word Similarity
The second task we run is word similarity calculation on pairs of words given in [SEMEVAL 2017](https://alt.qcri.org/semeval2017/task2/index.php?id=data-and-tools). Files and functionalities (all within `task_wordsim` directory):
- `task_wordsim/wordsimtrans.py`: translates English dataset into target language of choice. Usage:
```
python3 task_wordsim/wordsimtrans.py --word_file WORD_FILE --source_language --SOURCE_LANGUAGE --target_language TARGET_LANGUAGE
```
- `task_wordsim/wordsimftdatagen.py`: given a list of word pairs and similarity scores, generates a list of 3-dimensional vectors (Euclidean, Manhattan, and cosine distance between the words) as input into the word similarity task. Use for FastText-compatible model binaries only. Usage:
```
python3 task_wordsim/wordsimftdatgen.py --word_file WORD_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_wordsim/wordsimmusedatagen.py`: same functionality as `task_wordsim/wordsimftdatagen.py` but for MUSE-compatible models. Usage:
```
python3 task_wordsim/wordsimmusedatgen.py --word_file WORD_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_wordsim/wordsimtask.py`: runs the word similarity task given the data (3D vectors) file, the label file, and the model name. Outputs mean MSE loss over 30 trials. Usage:
```
python3 task_wordsim/wordsimtask.py --data_file DATA_FILE --label_file LABEL_FILE --model_name MODEL_NAME --num_trials NUM_TRIALS --dataset_size DATASET_SIZE --train_proportion TRAIN_PROPORTION --language LANGUAGE
```

### Bilingual Lexicon Induction
Lastly, we experiment on the cross-lingual tasks of bilingual lexicon induction both to and from English. Files and functionalities (all within `task_bli` directory):
- `task_bli/translationftdatagen.py`: given word pair training/testing files in both directions and model binaries, generates 300-dimensional embeddings of all the words (8 files total - 4-4 train-test split, 4-4 from-to split, 4-4 English-non-English split). Use for FastText-compatible model binaries only. Usage:
```
python3 task_bli/translationftdatagen.py --train_to_file TRAIN_TO_FILE --train_from_file TRAIN_FROM_FILE --test_to_file TEST_TO_FILE --test_from_file TEST_FROM_FILE --target_model_file TARGET_MODEL_FILE --english_model_file ENGLISH_MODEL_FILE --language LANGUAGE --model_name MODEL NAME
```
- `task_bli/translationmusedatagen.py`: same functionality as `task_bli/translationftdatagen.py` but for MUSE-compatible model binaries. Usage:
```
python3 task_bli/translationmusedatagen.py --train_to_file TRAIN_TO_FILE --train_from_file TRAIN_FROM_FILE --test_to_file TEST_TO_FILE --test_from_file TEST_FROM_FILE --target_model_file TARGET_MODEL_FILE --english_model_file ENGLISH_MODEL_FILE --language LANGUAGE --model_name MODEL NAME
```
- `task_bli/translationtask.py`: given vectorized data files, runs the BLI task in both directions and outputs mean cosine similarity as a performance metric for both directions. Usage:
```
python3 task_bli/translationtask.py --target_train_to_file TARGET_TRAIN_TO_FILE --english_train_to_file ENGLISH_TRAIN_TO_FILE --target_train_from_file TARGET_TRAIN_FROM_FILE --english_train_from_file ENGLISH_TRAIN_FROM_FILE --target_test_to_file TARGET_TEST_TO_FILE --english_test_to_file ENGLISH_TEST_TO_FILE --target_test_from_file TARGET_TEST_FROM_FILE --english_test_from_file ENGLISH_TEST_FROM_FILE --train_size TRAIN_SIZE --test_size TEST_SIZE --language LANGUAGe --model_name MODEL_NAME
```

We also provide some data files that can be used to run each of these code files with its default parameters. To run the files involving model binaries with our defaults, download FastText models from [here](https://fasttext.cc/docs/en/crawl-vectors.html) and MUSE models from [here](https://github.com/facebookresearch/MUSE#download).
