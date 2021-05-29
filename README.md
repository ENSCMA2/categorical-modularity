# Categorical Modularity: A Tool For Evaluating Word Embeddings

Categorical modularity is a low-resource intrinsic metric for evaluation of word embeddings. We provide code for the community to:
- Generate embedding vectors and nearest-neighbor matrices of lists of core words
- Calculate the following scores for said lists of words:
  - General categorical modularity with respect to a fixed list of semantic categories
  - Single-category modularity with respect to a fixed list of semantic categories
  - Network modularity of emerging categories from the nearest-neighbor graph created by a community detection algorithm
  
 ## Dependencies
 - [Python 3.6+](https://www.python.org/downloads/)
 - [scipy](https://www.scipy.org/)
 - [numpy](https://numpy.org/)
 - [nltk](https://www.nltk.org/)
 - [fasttext](https://fasttext.cc/)
 - [google_trans_new](https://pypi.org/project/google-trans-new/)
 - [scikit-learn](https://scikit-learn.org/stable/)
 - [networkx](https://networkx.org/)
 
 ## Calculating Modularity
Given a list of words and a `.bin` or `.vec` embedding model, you can calculate several modularity metrics using the files in the `core` directory. Brief descriptions of files and functionalities (further formatting specifications for files and parameters can be found by running the `--help` command on each file):
- `core/ft_vector_gen.py`: pass in a file containing a list of words and a file containing an embedding model binary and create a file with 300-dimensional embedding vectors of each of the input words. Use for FastText-compatible models (e.g. FastText, subs2vec). The lists of words and categories we used are labeled by language and level in `core/words`. Usage (with respect to topmost level directory of this repo): 
```
python3 core/ft_vector_gen.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUT_FILE
```
- `core/muse_vector_gen.py`: same functionality as `core/ft_vectorgen.py`, but used for MUSE-compatible models (a different implementation than the FastText library). Additionally asks for a language parameter specification for use in MUSE embedding generation in case a word requires stemming. This language parameter should be specified as the full lowercase name of the language (e.g. `english`, not `en`). Usage:
```
python3 core/muse_vector_gen.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUT_FILE --language LANGUAGE
```
- `core/ft_matrices.py`: given a list of n words and an embedding model binary, generates a file with an n x n matrix where row i, column j = k represents the fact that word j is the kth-nearest neighbor of word i in the given embedding space. Use this file for FastText-compatible models only. Usage:
```
python3 core/ft_matrices.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUTFILE
```
- `core/muse_matrices.py`: same functionality as `core/ft_matrices.py` but for MUSE-compatible model binaries. Usage: 
```
python3 core/muse_matrices.py --word_file WORD_FILE --model_file MODEL_FILE --out_file OUTFILE --language LANGUAGE
```
- `core/general_modularity.py`: calculate general categorical modularity given a list of categories and a matrix as generated by `core/ft_matrices.py` or `core/muse_matrices.py`, postprocessed to remove [] characters. Usage:
```
python3 core/general_modularity.py --categories_file CATEGORIES_FILE --matrix_file MATRIX_FILE
```
- `core/unsupervised_modularity.py`: calculate modularity of unsupervised clusters for all categories given a matrix as generated by `core/ft_matrices.py` or `core/muse_matrices.py`, postprocessed to remove [] characters. Usage:
```
python3 core/unsupervised_modularity.py --matrix_file MATRIX_FILE
```
- `core/correlation.py`: code that can be used to calculate the Spearman rank correlations of one set of modularity scores(`modularity_file`) with one set of task performance metrics (`downstream_file`). See `core/data` for default files - your inputs should conform to the formatting of these files. Usage:
```
python3 core/correlation.py --modularity_file MODULARITY_FILE --downstream_file DOWNSTREAM_FILE
```

## Single-Category Modularity Correlations
Our paper also explores an extension of categorical modularity to single-category modularity, which we test on each of the 59 categories listed in our paper. The `single_category` directory contains code that can be used to calculate these single-category modularities and their correlations with downstream task performance. Brief descriptions of files and functionalities:
- `single_category/single_category_correlation.py`: given a file with modularity scores for a set of categories and a file with performance metrics for a particular tasks, writes an output file with correlations between performance metrics and modularities with respect to each category. See `single_category/data/3_2.csv` for how the `modularity_file` should be formatted, and see `single_categories/movies_accuracy.csv` for how the `metrics_file` should be formatted. Usage:
```
python3 single_category/single_category_correlation.py --modularity_file MODULARITY_FILE --metrics_file METRICS_FILE --out_file OUT_FILE
```
- `single_category/single_category_modularity.py`: given a list of category labels for words and a square matrix of nearest neighbor relationships among words, calculates single-category modularities for each category. Examples for formatting purposes can be found in `single_category/data/categories_3.csv` (for `categories_file`) and `single_category/data/muse_finnish.csv` (for `matrix_file`). Usage: 
```
python3 single_category/single_category_modularity.py --categories_file CATEGORIES_FILE --matrix_file MATRIX_FILE
```


## Running Downstream Tasks
Our paper presents moderate to strong correlations of categorical modularity with four downstream tasks. We provide code to reproduce these tasks in the `task_bli`, `task_wordsim`, and `task_movies` directories.
### Sentiment Analysis
The first task we run is sentiment analysis of [IMDB movie reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). Files and functionalities:
- `task_movies/movie_data_gen.py`: given a file with data (raw text of movie reviews) and a target language, generates an equivalent dataset translated into the target language. The language name is the full English name of the language (e.g. `finnish`), while the language code is the 2-letter code (e.g. `fi`, full listing of codes [here](https://www.loc.gov/standards/iso639-2/php/code_list.php)). Usage:
```
python3 task_movies/movie_data_gen.py --data_file DATA_FILE --target_language_name TARGE_LANGUAGE_NAME --target_language_code TARGET_LANGUAGE_CODE
```
- `task_movies/ft_movie_gen.py`: given a data file with raw movie reviews and a model binary file, produces an output file 300-dimensional embeddings of each review. Use only for FastText-compatible models. Usage:
```
python3 task_movies/ft_movie_gen.py --data_file DATA_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_movies/muse_movie_gen.py`: given a data file with raw movie reviews and a model binary file, produces an output file 300-dimensional embeddings of each review. Use only for MUSE-compatible models. Usage:
```
python3 task_movies/muse_movie_gen.py --data_file DATA_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_movies/movie_task.py`: given data in the form of vectors (importantly, assuming the first half are positive and the second half are negative), runs the task of sentiment analysis and outputs mean accuracy and precision over 30 trials. Usage:
```
python3 task_movies/movie_task.py --data_file DATA_FILE --model_name MODEL_NAME --num_trials NUM_TRIALS --dataset_size DATASET_SIZE --train_proportion TRAIN_PROPORTION --language LANGUAGE
```

### Word Similarity
The second task we run is word similarity calculation on pairs of words given in [SEMEVAL 2017](https://alt.qcri.org/semeval2017/task2/index.php?id=data-and-tools). Files and functionalities (all within `task_wordsim` directory):
- `task_wordsim/wordsim_trans.py`: translates English dataset into target language of choice. Usage:
```
python3 task_wordsim/wordsim_trans.py --word_file WORD_FILE --source_language --SOURCE_LANGUAGE --target_language TARGET_LANGUAGE
```
- `task_wordsim/wordsim_ft_data_gen.py`: given a list of word pairs and similarity scores, generates a list of 3-dimensional vectors (Euclidean, Manhattan, and cosine distance between the words) as input into the word similarity task. Use for FastText-compatible model binaries only. Usage:
```
python3 task_wordsim/wordsim_ft_data_gen.py --word_file WORD_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_wordsim/wordsim_muse_data_gen.py`: same functionality as `task_wordsim/wordsim_ft_datagen.py` but for MUSE-compatible models. Usage:
```
python3 task_wordsim/wordsim_muse_data_gen.py --word_file WORD_FILE --model_file MODEL_FILE --model_name MODEL_NAME --language LANGUAGE
```
- `task_wordsim/wordsim_task.py`: runs the word similarity task given the data (3D vectors) file, the label file, and the model name. Outputs mean MSE loss over 30 trials. Usage:
```
python3 task_wordsim/wordsim_task.py --data_file DATA_FILE --label_file LABEL_FILE --model_name MODEL_NAME --num_trials NUM_TRIALS --dataset_size DATASET_SIZE --train_proportion TRAIN_PROPORTION --language LANGUAGE
```

### Bilingual Lexicon Induction
Lastly, we experiment on the cross-lingual tasks of bilingual lexicon induction both to and from English. Files and functionalities (all within `task_bli` directory):
- `task_bli/translation_ft_data_gen.py`: given word pair training/testing files in both directions and model binaries, generates 300-dimensional embeddings of all the words (8 files total - 4-4 train-test split, 4-4 from-to split, 4-4 English-non-English split). Use for FastText-compatible model binaries only. Usage:
```
python3 task_bli/translation_ft_data_gen.py --train_to_file TRAIN_TO_FILE --train_from_file TRAIN_FROM_FILE --test_to_file TEST_TO_FILE --test_from_file TEST_FROM_FILE --target_model_file TARGET_MODEL_FILE --english_model_file ENGLISH_MODEL_FILE --language LANGUAGE --model_name MODEL NAME
```
- `task_bli/translation_muse_data_gen.py`: same functionality as `task_bli/translation_ft_data_gen.py` but for MUSE-compatible model binaries. Usage:
```
python3 task_bli/translation_muse_data_gen.py --train_to_file TRAIN_TO_FILE --train_from_file TRAIN_FROM_FILE --test_to_file TEST_TO_FILE --test_from_file TEST_FROM_FILE --target_model_file TARGET_MODEL_FILE --english_model_file ENGLISH_MODEL_FILE --language LANGUAGE --model_name MODEL NAME
```
- `task_bli/translation_task.py`: given vectorized data files, runs the BLI task in both directions and outputs mean cosine similarity as a performance metric for both directions. Usage:
```
python3 task_bli/translation_task.py --target_train_to_file TARGET_TRAIN_TO_FILE --english_train_to_file ENGLISH_TRAIN_TO_FILE --target_train_from_file TARGET_TRAIN_FROM_FILE --english_train_from_file ENGLISH_TRAIN_FROM_FILE --target_test_to_file TARGET_TEST_TO_FILE --english_test_to_file ENGLISH_TEST_TO_FILE --target_test_from_file TARGET_TEST_FROM_FILE --english_test_from_file ENGLISH_TEST_FROM_FILE --train_size TRAIN_SIZE --test_size TEST_SIZE --language LANGUAGe --model_name MODEL_NAME
```

We also provide some data files that can be used to run each of these code files with its default parameters. To run the files involving model binaries with our defaults, download FastText models from [here](https://fasttext.cc/docs/en/crawl-vectors.html) and MUSE models from [here](https://github.com/facebookresearch/MUSE#download). Additionally, our paper discusses experiments on subs2vec embeddings, which can be found [here](https://github.com/jvparidon/subs2vec). 
