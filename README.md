# neural_note_linking
Categorize bits of text by how similar they are using neural language models

# Usage example

```python
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
# Since we are going to do several experiments, we pre-load the sentence transformer to save time
sentence_transformer = SentenceTransformer("distiluse-base-multilingual-cased")

dist = get_text_distances(texts, names, sentence_transformer, metric="l2")
```

```
cow1  cow3    0.771564
      cow2    0.805512
cow2  cow3    0.932953
it2   it3     1.169351
it1   it3     1.257774
      it2     1.273697
cow2  it1     1.280729
cow1  it1     1.290356
cow3  it1     1.308583
cow1  it3     1.332934
      it2     1.342553
cow2  it3     1.343927
cow3  it3     1.352675
cow2  it2     1.389026
cow3  it2     1.399683
```

The algorithm correctly sorts the texts by similarity, grouping all 3 texts about cows together and all 3 about IT together.
