import pickle

specialwords = [
    "mitarbeiter",
    "gesucht",
    "teilzeit",
    "vollzeit",
    "bereich",
    "ab",
    "std",
    "sofort",
    "r",
    "m",
    "w",
    "d",
    "x",
    "eur",
    "h",
    "voll",
    "basis",
    "div",
    "e",
    "c",
    "divers",
    "gn",
    "team",
    "suchen",
    "u",
    "sucht",
    "Job",
    "bzw",
]

with open("src/preprocessing/specialwords.tex", "wb") as fp:
    specialwords = pickle.dump(specialwords, fp)
