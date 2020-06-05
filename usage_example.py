from neural_note_linking import *

texts = [
    "I am a cow.",
    "My nature is that of a bovine.",
    "I live on a farm, am an animal, and give milk.",
    "Microprocessors are this company's specialty.",
    "Recent advances in quantum computing have been promising.",
    "Perhaps AI can save computational cost for businesses."
]

names = [
    "cow1",
    "cow2",
    "cow3",
    "it1",
    "it2",
    "it3"
]

tokenizer, model = get_tokenizer_and_model("xlm-roberta-base") # Recommended model for using this repo

dist_l2_mean = get_text_distances(texts, names, tokenizer, model, metric="l2", aggregation="mean")
print(dist_l2_mean.sort_values())

dist_l2_first = get_text_distances(texts, names, tokenizer, model, metric="l2", aggregation="first")
print(dist_l2_first.sort_values())
