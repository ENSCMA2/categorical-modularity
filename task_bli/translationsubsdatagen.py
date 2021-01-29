import fasttext
import sys

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']

train_to_files = ["TranslationData/" + i + '-en.0-5000.txt' for i in language_paths]
train_from_files = ["TranslationData/en-" + i + '.0-5000.txt' for i in language_paths]
test_to_files = ["TranslationData/" + i + '-en.5000-6500.txt' for i in language_paths]
test_from_files = ["TranslationData/en-" + i + '.5000-6500.txt' for i in language_paths]
i = int(sys.argv[1])

lang = langs[i]
print(lang)
model = fasttext.load_model('subs2vec/subs.' + language_paths[i] + '.bin')
englishmodel = fasttext.load_model('subs2vec/subs.en.bin')
with open(train_to_files[i], "r") as words:
  words = list(words)
  tgt_wds = []
  for w in words:
    tgt_wds.append(w.strip("\n").split()[0])
  eng_wds = []
  for w in words:
    eng_wds.append(w.strip("\n").split()[1])
  print(tgt_wds[:6])
  print(eng_wds[:6])
  tgt_vectors = [model[tgt_wds[i]] for i in range(len(tgt_wds))]
  eng_vectors = [englishmodel[eng_wds[i]] for i in range(len(eng_wds))]
  with open(lang + "to_english_" + lang + "_translationsubsvecstrain.txt", "w") as o:
    for v in tgt_vectors:
      o.write(str(list(v)) + "\n")
  with open(lang + "to_english_en_translationsubsvecstrain.txt", "w") as o:
    for v in eng_vectors:
      o.write(str(list(v)) + "\n")
print("done with train to")
with open(test_to_files[i], "r") as words:
  words = list(words)
  tgt_wds = [line.strip("\n").split()[0] for line in words]
  eng_wds = [line.strip("\n").split()[1] for line in words]
  print(tgt_wds[:6])
  print(eng_wds[:6])
  tgt_vectors = [model[tgt_wds[i]] for i in range(len(tgt_wds))]
  eng_vectors = [englishmodel[eng_wds[i]] for i in range(len(eng_wds))]
  with open(lang + "to_english_" + lang + "_translationsubsvecstest.txt", "w") as o:
    for v in tgt_vectors:
      o.write(str(list(v)) + "\n")
  with open(lang + "to_english_en_translationsubsvecstest.txt", "w") as o:
    for v in eng_vectors:
      o.write(str(list(v)) + "\n")
print("done with test to")
with open(train_from_files[i], "r") as words:
  words = list(words)
  tgt_wds = [line.strip("\n").split()[1] for line in words]
  eng_wds = [line.strip("\n").split()[0] for line in words]
  print(tgt_wds[:6])
  print(eng_wds[:6])
  tgt_vectors = [model[tgt_wds[i]] for i in range(len(tgt_wds))]
  eng_vectors = [englishmodel[eng_wds[i]] for i in range(len(eng_wds))]
  with open(lang + "from_english_" + lang + "_translationsubsvecstrain.txt", "w") as o:
    for v in tgt_vectors:
      o.write(str(list(v)) + "\n")
  with open(lang + "from_english_en_translationsubsvecstrain.txt", "w") as o:
    for v in eng_vectors:
      o.write(str(list(v)) + "\n")
print("done with train from")
with open(test_from_files[i], "r") as words:
  words = list(words)
  tgt_wds = [line.strip("\n").split()[1] for line in words]
  eng_wds = [line.strip("\n").split()[0] for line in words]
  print(tgt_wds[:6])
  print(eng_wds[:6])
  tgt_vectors = [model[tgt_wds[i]] for i in range(len(tgt_wds))]
  eng_vectors = [englishmodel[eng_wds[i]] for i in range(len(eng_wds))]
  with open(lang + "from_english_" + lang + "_translationsubsvecstest.txt", "w") as o:
    for v in tgt_vectors:
      o.write(str(list(v)) + "\n")
  with open(lang + "from_english_en_translationsubsvecstest.txt", "w") as o:
    for v in eng_vectors:
      o.write(str(list(v)) + "\n")
print("done with test from")
