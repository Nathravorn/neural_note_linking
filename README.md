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

# Since we are going to do several experiments, we pre-load the tokenizer and model to save time
tokenizer, model = get_tokenizer_and_model("xlm-roberta-base") # Recommended model for using this repo

dist_l2_mean = get_text_distances(texts, names, tokenizer, model, metric="l2", aggregation="mean")
print(dist_l2_mean.sort_values())
```

```
cow2  cow3    1.226743
it1   it2     1.320687
it2   it3     1.369428
cow3  it2     1.546401
it1   it3     1.552090
cow3  it1     1.564520
cow2  it3     1.586139
cow3  it3     1.589028
cow1  cow2    1.611313
cow2  it2     1.704291
      it1     1.739238
cow1  cow3    2.061035
      it3     2.335135
      it2     2.602877
      it1     2.644433
```

Clearly the model is not doing great. It rates a lot of cow and it texts as being closer than the two categories among themselves.

Let's try using the "first" aggregation function :

```python
dist_l2_first = get_text_distances(texts, names, tokenizer, model, metric="l2", aggregation="first")
print(dist_l2_first.sort_values())
```

```
it1   it2     0.766229
cow2  it1     0.854453
      it2     0.858894
cow1  it3     1.026549
      cow2    1.095686
it2   it3     1.133804
cow1  it2     1.152785
      it1     1.249897
      cow3    1.250869
it1   it3     1.255974
cow2  it3     1.322183
cow3  it3     1.351685
cow2  cow3    1.621630
cow3  it2     1.729994
      it1     1.776559
```

Welp, that's even worse. Sadface.
