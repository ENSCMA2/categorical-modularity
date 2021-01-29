import io
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import stopwords
import sys

l = int(sys.argv[1])
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
snowball = SnowballStemmer("porter")
sl = ['arabic', 'english', 'dutch', 'portuguese', 'spanish', 'danish', 'finnish',
'french', 'german', 'hungarian', 'italian', 'norwegian', 'romanian', 'russian',
'swedish']

def load_vec(emb_path, nmax=5000000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            if len(vect) != 300:
              continue
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

path_root = '/Users/karinahalevy/Code/MUSE/data/wiki.multi.'
path_suffix = '.vec'
language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi',
                  'fr', 'de', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']
languages = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french', 'german',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']
word_files = [i + '.txt' for i in languages]
vecs = [path_root + i + path_suffix for i in language_paths]
nmax = 5000000  # maximum number of word embeddings to load
stops = {'hebrew': ['אני', 'אני', 'שלי', 'את עצמי', 'אנחנו', 'שלנו', 'שלנו',
'את עצמנו', 'אתם', 'אתם', 'אתם', 'אתם', 'אתם', 'הייתם', 'שלכם', 'שלכם', 'עצמכם',
'עצמכם', 'הוא', 'הוא', 'שלו', 'עצמו', 'היא', 'היא', 'היא', 'שלה', 'עצמה', 'זה',
'זה', 'זה', 'זה', 'עצמו', 'הם', 'הם', 'שלהם', 'שלהם', 'שלהם', 'עצמם', 'מה',
'איזה', 'מי', 'מי', 'זה', 'זה', 'זה', 'זה', 'אלה', 'אלה', 'אני', 'הוא', 'היה', 'היה',
'עשה','א', 'ה', 'ו', 'אבל', 'אם', 'או', 'כי', 'כמו', 'עד',
'בעוד', 'של', 'ב', 'על ידי', 'עבור', 'עם', 'בערך', 'נגד', 'בין', 'אל', 'דרך', 'במהלך',
'לפני', 'אחרי', 'מעל', 'מתחת', 'עד', 'מ', 'למעלה', 'למטה', 'פנימה', 'על', 'כבוי',
'מעל', 'מעל', 'תחת', 'שוב', 'עוד יותר', 'ואז', 'פעם', 'כאן', 'שם', 'מתי', 'איפה',
'למה', 'איך', 'כולם', 'כל אחד', 'שניהם', 'כל אחד', 'כמה', 'יותר', 'יותר', 'אחר',
'כמה', 'כזה', 'לא', 'וגם', 'לא', 'רק', 'בבעלותו', 'אותו דבר', 'כך', 'מאשר יותר מדי',
'יכול', 'יכול', 'פשוט', 'לא', 'צריך', 'צריך', 'צריך', 'עכשיו', 'לא', 'לא יכול', 'לא',
'לא', 'לא', 'לא', 'לא', 'לא', 'לא', 'לא יכול', 'אסור', 'לא צריך', 'לא צריך', 'לא צריך',
'לא', 'לא', 'לא', 'לא', 'לא'],
'estonian': ['mina', 'minu', 'meie', 'ise', 'sina', 'sina ise', 'tema',
'tema ise', 'ta', 'ta on', 'tema', 'ta ise', 'see', 'see iseenesest', 'nemad',
'nende', 'oma', 'ise', 'mis', 'kes', 'see', 'need', 'need olen on', 'on', 'oli',
'olid', 'tuleb', 'on', 'mida', 'oli', 'millel teha', 'ei', 'tehes A-sse', 'on ja',
'aga kui või', 'sest kuna', 'kuni', 'samas', 'kus', 'juures', 'poolt', 'umbes',
'vastu', 'sisse', 'läbi', 'ajal', 'enne', 'pärast', 'üle', 'all', 'üles', 'alla',
'välja', 'üle', 'jälle', 'edasi', 'siis üks kord', 'siin', 'seal', 'millal',
'kus', 'miks', 'kuidas', 'kõiki', 'mõlemat', 'mõnd', 'vähe', 'enamikku', 'kõige',
'muud', 'mõnda', 'sellist', 'ei', 'ega', 'mitte', 'ainult', 'oma', 'sama', 'seega',
'ka', 'väga', 'saab', 'lihtsalt', 'ei peaks', 'peaks', 'oleks pidanud',
'nüüd ei peaks', 'ei saanud', 'ei', 'ei olnud', 'pole', 'ei ole', 'ei tohi',
'ei pea', 'pole', 'ei tohiks', 'polnud', 'ei olnud', 'ei oleks']}
print("on line 31")
loaded = [(load_vec(vecs[l], nmax))]
print("loaded all vectors")

def process_definition(definition, lang):
  try:
      sw = list(stopwords.words(lang))
  except:
      sw = stops[lang]
  words = [w.lower() for w in definition.split()]
  outwords = []
  for w in words:
    cleaned = ""
    for c in w:
      if c not in string.punctuation:
        cleaned += c
    if (cleaned not in sw):
      outwords.append(cleaned)
  return " ".join(outwords)

def average(embeddings):
    num_embeddings = len(embeddings)
    new_embedding = []
    for i in range(len(embeddings[0])):
        rs = 0
        for j in range(num_embeddings):
            rs += embeddings[j][i]
        new_embedding.append(rs / num_embeddings)
    return np.array(new_embedding)

def get_emb(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang, K=30):
    word2id = {v: k for k, v in src_id2word.items()}
    tok = word.split()
    embs = []
    for i in tok:
        try:
            e = src_emb[word2id[i]]
        except:
            try:
                e = src_emb[word2id[stemmer.stem(i)]]
            except:
                try:
                    e = src_emb[word2id[lemmatizer.lemmatize(i)]]
                except:
                    try:
                        e = src_emb[word2id[snowball.stem(i)]]
                    except:
                        try:
                            e = src_emb[word2id[SnowballStemmer(lang).stem(i)]]
                        except:
                            e = []
        if len(list(e)) > 0:
          embs.append(e)
    if len(embs) == 0:
        word_emb = np.array([0] * 300)
    else:
        word_emb = average(embs)
    return word_emb

def get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang):
    word2id = {v: k for k, v in src_id2word.items()}
    tok = word.split()
    embs = []
    for i in tok:
        try:
            e = src_emb[word2id[i]]
        except:
            try:
                e = src_emb[word2id[stemmer.stem(i)]]
            except:
                try:
                    e = src_emb[word2id[lemmatizer.lemmatize(i)]]
                except:
                    try:
                        e = src_emb[word2id[snowball.stem(i)]]
                    except:
                        try:
                            e = src_emb[word2id[SnowballStemmer(lang).stem(i)]]
                        except:
                            e = []
        if len(list(e)) > 0:
          embs.append(e)
    if len(embs) == 0:
        word_emb = np.array([0] * 300)
    else:
        word_emb = average(embs)
    # print("Nearest neighbors of \"%s\":" % word)
    scores = (tgt_emb / np.linalg.norm(tgt_emb, 2, 1)[:, None]).dot(word_emb / np.linalg.norm(word_emb))
    print(len(scores))
    print("ye")
    k_best = scores.argsort()[-1 * len(scores):][::-1]
    print(len(k_best))
    nns = []
    for i, idx in enumerate(k_best):
        nns.append((scores[idx], tgt_id2word[idx]))
    return nns

def get_nn_words(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang):
  nns = get_nn(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang)
  nn_w = []
  for s, w in nns:
    nn_w.append(w)
  return nn_w

def get_nn_word(word, words, src_emb, src_id2word, tgt_emb, tgt_id2word, lang):
  tr = []
  n = get_nn_words(word, src_emb, src_id2word, tgt_emb, tgt_id2word, lang)
  '''
  for w in words:
    try:
      tr.append(n.index(w))
    except:
      tr.append(-1)
  return tr
  '''

print(languages[l])
mat = []
with open(word_files[l], "r") as words:
  wds = [line.lower().strip("\n").strip(" ") for line in words]
  print(len(wds))
for j in range(1):
  vals = get_nn_word(wds[j], wds, loaded[0][0], loaded[0][1], loaded[0][0], loaded[0][1], languages[l])
  mat.append(vals)
'''
with open(languages[l] + "matrixfull500.txt", "w") as o:
  for q in range(len(wds)):
    o.write(str(mat[q]) + "\n")
print("done with" + languages[l])
'''

