import io
import numpy as np
import nltk
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem import WordNetLemmatizer
import math
from nltk.corpus import stopwords
import fasttext
import fasttext.util
import sys

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
snowball = SnowballStemmer("porter")
sl = ['arabic', 'english', 'dutch', 'portuguese', 'spanish', 'danish', 'finnish',
'french', 'german', 'hungarian', 'italian', 'norwegian', 'romanian', 'russian',
'swedish']

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

def get_nn_words(word, model):
  nns = []
  m = list(model.get_nearest_neighbors(word, k = 1000000))
  print(len(m))
  for s, w in m:
    nns.append(w)
  return nns

i = int(sys.argv[1])
print(languages[i])
mat = []
model = fasttext.load_model('wiki.' + language_paths[i] + '.bin')
with open(word_files[i], "r") as words:
  wds = [line.lower().strip("\n") for line in words]
  print(len(wds))
for j in range(1):
  neighbs = get_nn_words(wds[j], model)
  '''
  vals = []
  for n in range(1):
    try:
      k = neighbs.index(wds[n])
    except:
      k = -1
    vals.append(k)
  mat.append(vals)
  '''
'''
with open(languages[i] + "matrixft500.txt", "w") as o:
  for q in range(len(wds)):
    o.write(str(mat[q]) + "\n")
print("done with" + languages[i])
'''
