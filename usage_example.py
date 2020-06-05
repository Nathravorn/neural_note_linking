from neural_note_linking import get_text_distances

texts = [
    "I am a cow.",
    "My nature is that of a bovine.",
    "Je suis un animal qui donne du lait à ses veaux.",
    "Microprocessors are this company's specialty.",
    "Recent advances in quantum computing have been promising.",
    "La puissance des ordinateurs double tous les trois ans."
]

names = [
    "cow1",
    "cow2",
    "cow3",
    "it1",
    "it2",
    "it3"
]

dist = get_text_distances(texts, names)
print(dist.sort_values())
