import sys, subprocess, json, itertools, shutil
import pandas as pd
from neural_note_linking import get_text_distances


# A list of zettels along with connectivity information
class ZettelBag():
    def __init__(self, zettel_dir):
        self.dir = zettel_dir
        command = ["neuron", "-d", zettel_dir, "query", "--graph"]
        graph_raw = subprocess.run(command, capture_output=True).stdout
        self.graph = json.loads(graph_raw)["result"]
        self.ids = list(self.graph["vertices"])

    # Iterate over the zettels in this bag.
    def zettels(self):
        return (self.get(id) for id in self.ids)

    # Return all pairs of zettels that have a link between them.
    def adjacent_pairs(self):
        adjacencyMap = self.graph["adjacencyMap"]
        links = set(
            (src, tgt)
            for (src, tgts) in adjacencyMap.items()
            for tgt in tgts
        )
        links = links.union((y, x) for (x, y) in links)
        return links

    # Return the zettel corresponding to the given id
    def get(self, id):
        return Zettel(self, id)

    # Remove zettels that match the predicate from the bag
    def drop_zettels(self, predicate):
        self.ids = [ zet.id for zet in self.zettels() if not predicate(zet) ]

    # Remove zettels that have tags we don't want
    def exclude_tags(self, tags):
        tags = set(tags)
        self.drop_zettels(lambda zet: bool(tags.intersection(zet.tags)))

    # Return a list of pairs of zettels that are semantically close, along with
    # a closeness score. The list is sorted with closest first.
    def closest_linkable_pairs(self):
        texts = [ zet.read_contents() for zet in self.zettels() ]
        ids = [ zet.id for zet in self.zettels() ]
        # Calculate all pairwise distances between zettels
        srs =  get_text_distances(texts, ids)

        # Remove pairs of zettels that have a link
        to_exclude = self.adjacent_pairs().intersection(srs.index)
        srs = srs.drop(to_exclude)
        srs = srs.sort_values()

        return (
            ((self.get(a), self.get(b)), score)
            for (a, b), score in srs.iteritems()
        )

# A zettel
class Zettel():
    def __init__(self, bag, id):
        self.bag = bag
        self.id = id
        properties = bag.graph["vertices"][id]
        self.title = properties["zettelTitle"]
        self.tags = properties["zettelTags"]
        self.properties = properties

    # Read the contents of the zettel from the filesystem.
    def read_contents(self):
        file = f"{self.bag.dir}/{self.id}.md"
        return open(file).read()

# Pretty-print the output of `closest_linkable_pairs` in a table
def display_pairs(pairs):
    cols = shutil.get_terminal_size((80, 20)).columns
    w = int((cols - len("{0:.5f}") - 2) / 2)
    truncate_and_pad = lambda s: f"{{:<{w}}}".format(s[:w])
    def print_row(x, y, s):
        print(
            truncate_and_pad(x),
            truncate_and_pad(y),
            s,
            sep="│"
        )

    print_row("Zettel 1", "Zettel 2", "Score")
    print("─" * w, "─" * w, "─" * (cols-2*w-2), sep="┼")
    mktitle = lambda zet: f"[{zet.id}] {zet.title}"
    for (a, b), score in pairs:
        a = mktitle(a)
        b = mktitle(b)
        print_row(a, b, f"{score:.5f}")


if __name__ == "__main__":
    zettel_dir = sys.argv[1]
    bag = ZettelBag(zettel_dir)
    items = bag.closest_linkable_pairs()
    best_n_items = itertools.islice(items, 50)
    display_pairs(best_n_items)
