import fasttext
import sys
from google_trans_new import google_translator

langs = ['english', 'french', 'spanish', 'italian']

language_paths = ['en', 'fr', 'es', 'it']

extra_langs = ['arabic', 'vietnamese', 'bulgarian', 'hebrew', 'croatian', 'estonian',
'czech', 'danish', 'finnish', 'greek', 'hungarian', 'indonesian',
              'macedonian', 'norwegian', 'polish', 'romanian',
              'russian', 'slovak', 'slovenian', 'swedish',
              'turkish', 'ukrainian', 'portuguese', 'dutch', 'catalan', 'french']

extra_paths = ["ar", "vi", "bg", "he", "hr", "et", "cs", "da", "fi", "el",
"hu", "id", "mk", "no", "pl", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
"pt", "nl", "ca", "fr"]

i = int(sys.argv[1])
j = int(sys.argv[2])
print(langs[i])
print(extra_langs[j])
translator = google_translator()
txtfile = language_paths[i].upper() + "_SEMEVAL17.txt"

with open(txtfile, "r+") as o:
  words = [line.strip("\n").split("\t") for line in list(o)]
  p1 = []
  p2 = []
  s = []
  for w in words:
    w1 = translator.translate(w[0], lang_src = language_paths[i],
      lang_tgt = extra_paths[j])
    w2 = translator.translate(w[1], lang_src = language_paths[i],
      lang_tgt = extra_paths[j])
    p1.append(w1)
    p2.append(w2)
    print(w1 + "\t" + w2 + "\t" + w[2])
    s.append(w[2])
  with open(extra_paths[j].upper() + "_SEMEVAL17.txt", "w") as tw:
    for k in range(len(p1)):
      tw.write(p1[k] + "\t" + p2[k] + "\t" + s[k] + "\n")
