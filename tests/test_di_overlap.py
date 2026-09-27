"""The Decision Index row filter: suite items out, shared boilerplate in."""
import numpy as np

from lod.corpus.services.decontaminate import Blocklist, item_hashes
from lod.schema import Example, Question

ITEM = ("Which of the following best describes the primary function of the enzyme "
        "helicase during the replication of a double stranded DNA molecule in cells")
SHORT = "how do i reset my online banking password"


def _bl(*texts):
    long, item, size, short = [], [], [], []
    for t in texts:
        lo, sh = item_hashes(t)
        lo = sorted(set(lo))
        if lo:
            long += lo; item += [len(size)] * len(lo); size.append(len(lo))
        short += sh
    return Blocklist(np.array(long), np.array(item), np.array(size), np.array(short))


def _ex(state):
    return Example(state=state, task="t", questions=[Question("q", "?", ["a", "b"], 0)])


def test_an_embedded_suite_item_is_dropped():
    b = _bl(ITEM, SHORT)
    assert b.blocked("Answer the question.\n" + ITEM + "\nA) x B) y")
    assert b.blocked(SHORT)                              # a whole short item
    assert b.blocked("Intent example:\n" + SHORT.upper() + "\n")


def test_a_fragment_or_short_line_is_not():
    b = _bl(ITEM, SHORT)
    half = " ".join(ITEM.split()[:9])                    # 2 shingles of 16
    assert not b.blocked("unrelated text " + half + " and more words here")
    assert not b.blocked("reset my password")            # 3 words, not the item
    kept, n = b.filter([_ex(ITEM), _ex("nothing to see")])
    assert n == 1 and kept[0].state == "nothing to see"


def test_non_latin_text_is_matched():
    """`[a-z0-9]+` saw no token in Arabic text, so an Arabic suite item could never match."""
    ar = ("هذا مثال طويل على تغريدة عربية ساخرة جدا عن الطقس اليوم في المدينة الكبيرة "
          "والناس في الشوارع")
    b = _bl(ar)
    assert b.blocked("tweet:\n" + ar)
    assert not b.blocked("نص مختلف تماما لا علاقة له بالتغريدة الأصلية أبدا")
