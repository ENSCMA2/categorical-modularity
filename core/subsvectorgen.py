import fasttext

langs = ['arabic', 'bulgarian', 'catalan', 'croatian', 'czech', 'danish',
              'dutch', 'english', 'estonian', 'finnish', 'french',
              'greek', 'hebrew', 'hungarian', 'indonesian', 'italian',
              'macedonian', 'norwegian', 'polish', 'portuguese', 'romanian',
              'russian', 'slovak', 'slovenian', 'spanish', 'swedish',
              'turkish', 'ukrainian', 'vietnamese']

language_paths = ['ar', 'bg', 'ca', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi',
                  'fr', 'el', 'he', 'hu', 'id', 'it', 'mk', 'no', 'pl',
                  'pt', 'ro', 'ru', 'sk', 'sl', 'es', 'sv', 'tr', 'uk', 'vi']

word_files = [i + '.txt' for i in langs]
for i in range(len(langs)):
  lang = langs[i]
  print(lang)
  model = fasttext.load_model('subs2vec/subs.' + language_paths[i] + '.bin')
  with open(word_files[i], "r") as words:
    wds = [line.lower().strip("\n") for line in words]
    print(len(wds))
    vectors = [model[wds[i]] for i in range(len(wds))]
  with open(lang + "_subsvecs.txt", "w") as o:
    for v in vectors:
      o.write(str(list(v)) + "\n")
