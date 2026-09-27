"""Row 49 -- many-option matching and linking (domain 23).

Almost every question in the corpus has 2-10 options. This row is where a question has
two hundred, and where the wrong options are *almost* right. Four families, one engine:

    match   a request (customer message, spec sheet, purchase-order line) against catalog
            entries -- products, parts, API endpoints, plans -- each a code and a short
            description
    link    a passage mentioning an entity against knowledge-base candidates that all share
            its surface name
    nav     a goal and the page the agent is on against the links it could follow
    route   a ticket against the queues it could be routed to

**Code owns the target.** Every entry is a vector of attribute values; the state states k
constraints (a threshold, an exclusion, an either/or, a qualifier wanted or not wanted)
over k of the domain's attributes; exactly one entry satisfies all of them, and `build`
asserts it by evaluating every entry, not by trusting how it was drawn.

**The option set is a grid, not a star.** The entries are a uniform random subset of a
Cartesian grid over the constrained attributes (one satisfying value per axis), dealt into
*blocks* that share their unconstrained attributes (a product line, a set of namesakes,
one site section), and the gold is a uniformly random entry of them. A **hard negative** is a grid cell one edit from the gold -- a size, a version, a
date, a unit, a near-synonym (`water-resistant` for `waterproof`), a qualifier dropped or
added -- so it fails exactly one stated constraint. The obvious way to make hard negatives,
edit the gold k ways and add random fillers, builds a star whose centre is the answer:
"the option most like the others" then scores 1.000 whatever N is. On a full grid every
cell has the same neighbourhood, and a uniform subsample keeps that true in expectation, so
the medoid is the gold at 1/N (measured by a shortcut audit). The price is the hard-
negative share: the one-edit share of a grid is sum(|V_j| - 1) / prod(|V_j|), and only
the gold's block-mates count, so it is 30 % of distractors at N <= 10, 17 % at 11-100 and
9 % past 100 (19 % overall; entries failing exactly one constraint, other blocks' near
misses included, are 27 %, `meta.gen.near_share`). More hard negatives at large N would have to cluster around
the gold, and that cluster is the shortcut.

Only after the gold is drawn are its coordinates mapped to the satisfying values, so it is
exchangeable with every other entry in whatever a reader could count: how often its
values recur, how large its block is, how many entries are one edit away (see `build` for
the two versions that were not).

**No literal match decides.** The first version of this row stated every constraint in the
options' own words, so "the entry sharing the most words with the request" scored 0.40-0.52
against a 0.10 chance. Now, per axis, *which* of the axis's values the state names
literally is exchangeable between the gold and every other value on the axis: with
probability MU the state names one value, drawn uniformly -- the gold's (a literal
requirement: "in navy", "at least 30 L" against a 30 L entry) or a failing one (the
requirement said another way: an alias, "in German" against `/de/`, a threshold no entry
prints, and the failing value mentioned in passing: "my old one was 25 L"; or the
exclusion itself: "anything but black", a strict "over 30 L" against a 30 L entry). An
attribute with no alias for its values names *all* of the axis's values (the requirement,
and the others as ones already tried), which is exchangeable too. A reader counting shared
words is then at chance.

**No single constraint decides.** k is 2-4, every axis of the gold's block carries failing
values, and which axis another block breaks is uniform, so applying one constraint and
guessing among the survivors is well short of the rule.

**Held out by domain.** Each family trains on three or four domains (catalogs, KB types,
site layouts, organisations); one more is the dev split's and one the test split's, and
the three sets are disjoint. `_eval` twins of the trained domains (fresh seed) go to
`devreal` as in-structure evaluation. `catalog_<family>_largeN` (N 200-2,000, test domain
only) are for sharded scoring: they do not fit the packer and carry `meta.large_n`.

**Option count** is log-uniform over [2, 512], then cut to what fits the training budget
(state <= 2,304, total <= 3,072 Qwen3 tokens): at most ~165 options, and in the built
train split 30 % N 2-10, 48 % 11-100, 22 % 101-165. The packer keeps all of it whole.

`meta.gen` records the draw (family, domain, N, k, hard-negative share, blocks, grid shape,
state style, constrained attributes, ops and mention cases) for error-directed generation;
`meta.hard_neg` lists the hard negatives' option indices and `meta.hn_kind` their edit kind.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from lod.schema import Example, Question
from lod.corpus.services.sources.real.base import RealTask

LICENCE = "generated by lod/corpus/services/sources/synth/catalog.py; no third-party data"
URL = "https://github.com/(this repo)/blob/main/src/lod/corpus/services/sources/synth/catalog.py"
ROW = 49
DOMAIN = 23

N_MIN, N_MAX = 2, 512
LARGE_MIN, LARGE_MAX = 200, 2000
MAX_STATE_TOKENS = 2304
MAX_TOTAL_TOKENS = 3072
# A source module does not load a tokenizer (see browser.py), so the budget is estimated:
# Qwen3 spends one token per digit, and the rest of this text runs ~3 characters a token.
# `est_tokens` counts digits alone and the other characters at the domain's `cpt`, set
# below the lowest ratio measured on that domain with Qwen/Qwen3-0.6B-Base at full budget
# (2.0 for the docs site's short URLs, 3.7 for university tickets). The first build, with
# constants from small samples, lost 2 of 23,940 train examples to the packer (docs site);
# `test_every_example_fits_the_training_budget` holds it to the real tokenizer.
FIT_TOKENS = 2980                    # of 3,072: headroom for the estimate's error

P_ONE_BLOCK = 0.55                   # the gold's block is the whole option set
MAX_BLOCKS = 5
MU = 0.6                             # per axis: the state names one of the axis's values
NOALIAS_CAP = 4                      # values on an axis whose values have no alias
P_UNIT_FLIP = 0.6                    # include the gold's number in the other unit
K_WEIGHTS = ((2, 0.40), (3, 0.40), (4, 0.20))
# Large N needs a grid of as many distinct combinations as entries, so more axes.
K_WEIGHTS_LARGE = ((3, 0.40), (4, 0.60))

# Per-task volumes. generate hands every task of a row the same quota; the loaders
# stop here, which sets the split sizes the build asks for: 14 x 1,800 train, 14 x 72
# in-structure twins, 4 x 625 held-out dev, 4 x 625 held-out test, 4 x 75 large-N.
TRAIN_CAP = 1800
TWIN_CAP = 72
HELDOUT_CAP = 625
LARGE_CAP = 75


# ---- attributes and domains ------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Attr:
    name: str
    label: str
    kind: str                        # "num" | "enum" | "flag"
    values: tuple = ()
    say: str = "{x}"                 # clause fragment; {x} the stated content, {ax} with a/an
    neg: str | None = None           # enum "ne" clause, {x} the excluded value
    fmt: str = "{n}"                 # num: rendering in the option
    alt: tuple = ()                  # num: ((unit, factor to base, numbers, fmt), ...)
    ops: tuple = ()
    scale: str = "qty"               # num: "qty" | "time" | "ver"
    syn: tuple = ()                  # enum: near-synonym groups
    # enum: ((value, wording), ...) naming the value in no word the option uses. With
    # `alias_clause` the wording is a whole clause ("played cricket"), else it fills {x}.
    alias: tuple = ()
    alias_clause: bool = False
    text: str = ""                   # flag: rendered in the option when True
    need: str = ""
    avoid: str = ""
    show: Callable | None = None     # num: n -> option text (overrides fmt)
    tshow: Callable | None = None    # num: n -> text in the state (defaults to show)
    decades: bool = False            # num/time: "the early 1970s" style ranges


@dataclass(frozen=True, eq=False)
class Domain:
    name: str
    family: str
    noun: str
    attrs: tuple
    desc: str                        # option description template ("" for none)
    key: str = "sku"                 # "sku" | "entity" | "url" | "queue"
    url: str = ""                    # nav: the option key template
    names: tuple = ()                # link: surface names
    intro: tuple = ()                # family-specific opening lines
    sub: str = ""                    # link: how the passage refers back to the entity
    cpt: float = 2.0                 # non-digit characters per Qwen3 token, see est_tokens

    def attr(self, name: str) -> Attr:
        return next(a for a in self.attrs if a.name == name)


def num(name, label, values, **kw) -> Attr:
    return Attr(name, label, "num", tuple(values), **kw)


def enum(name, label, values, **kw) -> Attr:
    return Attr(name, label, "enum", tuple(values), **kw)


def flag(name, label, text, need, avoid, **kw) -> Attr:
    return Attr(name, label, "flag", (False, True), text=text, need=need, avoid=avoid,
                ops=("need", "avoid"), **kw)


QTY_OPS = (">=", "<=", ">", "<", "range")
TIME_OPS = (">=", "<=", ">", "<", "range", "eq")
ENUM_OPS = ("eq", "ne", "in")

_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")


def _k(n):
    return f"{n / 1000:g}k" if n >= 1000 else str(n)


def _ver(n):
    return f"{n // 10}.{n % 10}"


def _ym_url(n):
    return f"{n // 12}/{n % 12 + 1:02d}"


def _ym_text(n):
    return f"{_MONTHS[n % 12]} {n // 12}"


def _ym(y0, y1):
    return tuple(y * 12 + m for y in range(y0, y1 + 1) for m in range(0, 12, 2))


# -- match ---------------------------------------------------------------------------------
BACKPACKS = Domain(
    "backpacks", "match", "backpack", (
        num("capacity", "capacity", (10, 12, 14, 15, 16, 18, 20, 22, 24, 25, 26, 28, 30, 32,
                                     34, 35, 38, 40, 42, 45, 48, 50, 55, 60, 65, 70),
            fmt="{n} L", say="with a capacity of {x}", ops=QTY_OPS),
        num("weight", "weight", (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8,
                                 2.0, 2.2, 2.4),
            fmt="{n} kg", alt=(("lb", 0.4536, (1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0,
                                               3.3, 3.6, 4.0, 4.4, 4.8), "{n} lb"),),
            say="weighing {x}", ops=QTY_OPS),
        enum("proofing", "weather rating", ("waterproof", "water-resistant", "water-repellent",
                                            "splash-proof", "weatherproof"),
             syn=(("waterproof", "water-resistant", "water-repellent", "splash-proof",
                   "weatherproof"),),
             say="that is {x}", ops=("eq", "ne")),
        enum("colour", "colour", ("black", "charcoal", "grey", "navy", "teal", "olive", "khaki",
                                  "sand", "red", "burgundy", "orange", "mustard"),
             syn=(("black", "charcoal", "grey"), ("navy", "teal"), ("olive", "khaki", "sand"),
                  ("red", "burgundy", "orange")),
             say="in {x}", ops=ENUM_OPS),
        flag("sleeve", "laptop sleeve", ", laptop sleeve", "with a laptop sleeve",
             "without a laptop sleeve"),
    ),
    "{colour}, {capacity}, {weight}, {proofing}{sleeve}", cpt=2.05)

FASTENERS = Domain(
    "fasteners", "match", "bolt", (
        enum("thread", "thread", ("M3", "M4", "M5", "M6", "M8", "M10", "M12", "M14", "M16",
                                  "M20"),
             say="with {x} thread", neg="with any thread but {x}", ops=("eq", "in", "ne")),
        num("length", "length", (6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70,
                                 80, 90, 100, 110, 120),
            fmt="{n} mm", say="{x} long", ops=QTY_OPS),
        enum("material", "material", ("A2 stainless", "A4 stainless", "zinc-plated",
                                      "galvanised", "brass", "titanium", "nylon",
                                      "black oxide"),
             syn=(("A2 stainless", "A4 stainless"), ("zinc-plated", "galvanised",
                                                     "black oxide")),
             say="in {x}", ops=ENUM_OPS),
        enum("head", "head", ("hex head", "socket cap", "pan head", "countersunk",
                              "button head", "flange head", "cheese head"),
             syn=(("pan head", "button head", "cheese head"), ("hex head", "flange head")),
             say="with {ax}", neg="with anything but {ax}", ops=ENUM_OPS),
        flag("fine", "fine pitch", ", fine pitch", "with a fine-pitch thread",
             "with a coarse rather than fine pitch"),
    ),
    "{thread} x {length} {head}, {material}{fine}", cpt=2.35)

LAPTOPS = Domain(
    "laptops", "match", "laptop", (
        num("screen", "screen", (11.6, 12.4, 13.3, 13.6, 14.0, 14.5, 15.6, 16.0, 17.3),
            fmt="{n}\"", say="with a screen of {x}", ops=QTY_OPS + ("eq",)),
        num("ram", "memory", (4, 8, 12, 16, 24, 32, 48, 64, 96),
            fmt="{n} GB RAM", say="with {x} of memory", ops=QTY_OPS + ("eq",)),
        enum("os", "operating system", ("Windows Home", "Windows Pro", "ChromeOS", "Ubuntu",
                                        "Fedora", "macOS"),
             syn=(("Windows Home", "Windows Pro"), ("Ubuntu", "Fedora")),
             say="running {x}", ops=ENUM_OPS),
        enum("keys", "keyboard", ("UK", "US", "German", "French", "Nordic", "Spanish",
                                  "Swiss", "Japanese"),
             syn=(("UK", "US"), ("German", "Swiss"), ("French", "Spanish")),
             alias=(("UK", "British"), ("US", "American"), ("German", "Deutsch"),
                    ("French", "AZERTY"), ("Nordic", "Scandinavian"),
                    ("Spanish", "Castilian"), ("Swiss", "Helvetic"),
                    ("Japanese", "JIS")),
             say="with a {x} keyboard layout", ops=("eq", "in")),
        flag("touch", "touchscreen", ", touchscreen", "with a touchscreen",
             "without a touchscreen"),
    ),
    "{screen}, {ram}, {os}, {keys} keys{touch}", cpt=2.2)

API_ENDPOINTS = Domain(
    "api_endpoints", "match", "API endpoint", (
        enum("method", "method", ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"),
             syn=(("PUT", "PATCH", "POST"), ("GET", "HEAD")),
             say="using {x}", ops=("eq", "in")),
        enum("resource", "resource", ("invoices", "customers", "subscriptions", "refunds",
                                      "payouts", "disputes", "webhooks", "products", "orders",
                                      "coupons", "credit-notes", "quotes"),
             syn=(("invoices", "credit-notes", "quotes"), ("refunds", "payouts", "disputes"),
                  ("products", "orders")),
             alias=(("invoices", "bills we send"), ("customers", "client accounts"),
                    ("subscriptions", "recurring plans"), ("refunds", "money given back"),
                    ("payouts", "bank transfers to merchants"),
                    ("disputes", "chargeback cases"), ("webhooks", "event callbacks"),
                    ("products", "catalogue items"), ("orders", "purchases"),
                    ("coupons", "discount codes"), ("credit-notes", "credit memos"),
                    ("quotes", "price estimates")),
             say="for {x}", ops=("eq", "in")),
        num("version", "API version", (1, 2, 3, 4, 5, 6, 7), fmt="v{n}",
            say="on API {x}", ops=(">=", "<=", "eq")),
        enum("scope", "scope", ("read-only", "read-write", "admin", "public"),
             syn=(("read-only", "read-write"), ("admin", "public")),
             say="with {x} scope", ops=("eq", "ne")),
        num("rate", "rate limit", (10, 20, 30, 50, 60, 100, 120, 200, 300, 500, 1000),
            fmt="{n} req/min", say="allowing {x}", ops=QTY_OPS),
        flag("paged", "pagination", ", paginated", "that returns paginated results",
             "that returns results that are not paginated"),
    ),
    "{method} /{version}/{resource}, {scope}, {rate}{paged}", cpt=2.3)

TYRES = Domain(
    "tyres", "match", "tyre", (
        num("width", "width", (165, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265, 275),
            fmt="{n}", say="{x} mm wide", ops=QTY_OPS + ("eq",)),
        num("profile", "profile", (35, 40, 45, 50, 55, 60, 65, 70),
            say="with a profile of {x}", ops=("eq", "range")),
        num("rim", "rim", (14, 15, 16, 17, 18, 19, 20, 21),
            say="for a {x} inch rim", ops=("eq", "range")),
        enum("season", "season", ("summer", "winter", "all-season", "all-weather", "studded"),
             syn=(("all-season", "all-weather"), ("winter", "studded")),
             say="for {x} use", ops=ENUM_OPS),
        flag("runflat", "run-flat", ", run-flat", "that is run-flat", "that is not run-flat"),
    ),
    "{width}/{profile} R{rim}, {season}{runflat}", cpt=1.95)

CLOUD_PLANS = Domain(
    "cloud_plans", "match", "server plan", (
        num("vcpu", "vCPUs", (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64), fmt="{n} vCPU",
            say="with {x}", ops=QTY_OPS + ("eq",)),
        num("memory", "memory", (1, 2, 4, 8, 12, 16, 24, 32, 64, 128, 256), fmt="{n} GB",
            say="with {x} of memory", ops=QTY_OPS + ("eq",)),
        enum("region", "region", ("us-east", "us-west", "eu-west", "eu-central", "eu-north",
                                  "ap-south", "ap-northeast", "sa-east", "ca-central"),
             syn=(("us-east", "us-west"), ("eu-west", "eu-central", "eu-north"),
                  ("ap-south", "ap-northeast")),
             alias=(("us-east", "Virginia"), ("us-west", "Oregon"), ("eu-west", "Ireland"),
                    ("eu-central", "Frankfurt"), ("eu-north", "Stockholm"),
                    ("ap-south", "Mumbai"), ("ap-northeast", "Tokyo"),
                    ("sa-east", "Sao Paulo"), ("ca-central", "Montreal")),
             say="hosted in {x}", ops=("eq", "in")),
        num("price", "price", (5, 8, 10, 12, 15, 20, 24, 30, 40, 48, 60, 80, 96, 120, 160),
            fmt="${n}", say="costing {x} a month", ops=QTY_OPS),
        flag("backups", "backups", ", backups", "with daily backups",
             "without backups"),
    ),
    "{vcpu}, {memory}, {region}, {price}{backups}", cpt=1.8)

# -- link ----------------------------------------------------------------------------------
PERSON = Domain(
    "person", "link", "person", (
        enum("occupation", "occupation", ("footballer", "cricketer", "rugby player",
                                          "politician", "painter", "sculptor", "novelist",
                                          "poet", "chemist", "physicist", "architect",
                                          "cyclist", "bishop", "jazz pianist", "organist"),
             syn=(("footballer", "rugby player", "cricketer"), ("painter", "sculptor"),
                  ("novelist", "poet"), ("chemist", "physicist"),
                  ("jazz pianist", "organist")),
             alias=(("footballer", "played professional football"),
                    ("cricketer", "played county cricket"),
                    ("rugby player", "played in the scrum"),
                    ("politician", "sat in parliament"), ("painter", "painted in oils"),
                    ("sculptor", "carved stone figures"), ("novelist", "wrote novels"),
                    ("poet", "wrote verse"), ("chemist", "did research in chemistry"),
                    ("physicist", "did research in physics"),
                    ("architect", "designed buildings"), ("cyclist", "raced bicycles"),
                    ("bishop", "led a diocese"),
                    ("jazz pianist", "played piano in nightclubs"),
                    ("organist", "played the cathedral organ")),
             alias_clause=True, say="worked as {ax}", neg="never worked as {ax}",
             ops=ENUM_OPS),
        enum("nationality", "nationality", ("Welsh", "Scottish", "English", "Irish",
                                            "Australian", "Canadian", "New Zealand",
                                            "South African", "American"),
             syn=(("Welsh", "Scottish", "English", "Irish"),
                  ("Australian", "New Zealand"), ("Canadian", "American")),
             alias=(("Welsh", "comes from Wales"), ("Scottish", "comes from Scotland"),
                    ("English", "comes from England"), ("Irish", "comes from Ireland"),
                    ("Australian", "comes from Australia"), ("Canadian", "comes from Canada"),
                    ("New Zealand", "is a Kiwi"), ("South African", "grew up near the Cape"),
                    ("American", "comes from the United States")),
             alias_clause=True, say="is {x}", neg="is not {x}", ops=ENUM_OPS),
        num("born", "birth year", tuple(range(1890, 2004)), say="was born {x}",
            scale="time", ops=TIME_OPS, decades=True),
        enum("city", "city", ("Cardiff", "Swansea", "Glasgow", "Leeds", "Bristol", "Dublin",
                              "Belfast", "Melbourne", "Toronto", "Boston", "Auckland",
                              "Cape Town", "Manchester"),
             say="was based in {x}", neg="was never based in {x}", ops=ENUM_OPS),
        flag("knighted", "knighthood", ", knighted", "was knighted", "was never knighted"),
    ),
    "{nationality} {occupation}, b. {born}, {city}{knighted}",
    key="entity",
    names=("John Smith", "David Jones", "Maria Garcia", "James Brown", "Sarah Wilson",
           "Peter Hughes", "Karen Davies", "Michael Evans", "Paul Roberts", "Anne Walker",
           "Thomas Wright", "Helen Morgan"),
    intro=("{name} was mentioned in this week's regional news.",
           "A reader asked the archive about {name}.",
           "The collection holds a letter signed by {name}.",
           "{name} appears in the minutes of an old meeting."),
    sub="This {name}", cpt=2.9)

SETTLEMENT = Domain(
    "settlement", "link", "place", (
        enum("type", "type", ("city", "town", "village", "hamlet", "borough", "parish",
                              "suburb"),
             syn=(("town", "borough"), ("village", "hamlet", "parish"), ("city", "suburb")),
             say="is {ax}", neg="is not {ax}", ops=ENUM_OPS),
        enum("country", "country", ("England", "Wales", "Scotland", "Ireland", "Canada",
                                    "the United States", "Australia", "New Zealand",
                                    "Jamaica", "South Africa"),
             alias=(("England", "is English"), ("Wales", "is Welsh"),
                    ("Scotland", "is Scottish"), ("Ireland", "is Irish"),
                    ("Canada", "is Canadian"), ("the United States", "is American"),
                    ("Australia", "is Australian"), ("New Zealand", "is a Kiwi place"),
                    ("Jamaica", "is Jamaican"), ("South Africa", "is near the Cape")),
             alias_clause=True, say="lies in {x}", neg="does not lie in {x}", ops=ENUM_OPS),
        num("population", "population", (120, 300, 650, 900, 1800, 3500, 6200, 12000, 24000,
                                         48000, 95000, 180000, 350000, 700000),
            show=_k, say="has a population of {x}", ops=QTY_OPS),
        enum("river", "river", ("Usk", "Thames", "Severn", "Tyne", "Avon", "Mersey", "Ouse",
                                "Trent", "Clyde", "Shannon", "Hudson", "Murray"),
             say="stands on the {x}", neg="does not stand on the {x}", ops=ENUM_OPS),
    ),
    "{type} in {country}, pop. {population}, on the {river}",
    key="entity",
    names=("Newport", "Richmond", "Springfield", "Kingston", "Milton", "Ashford", "Clifton",
           "Fairview", "Georgetown", "Weston", "Bradford", "Halton"),
    intro=("The 1911 gazetteer has an entry for {name}.",
           "A traveller's diary describes a stop at {name}.",
           "The parcel was addressed to {name}.",
           "{name} is named in a planning notice."),
    sub="This {name}", cpt=2.5)

COMPANY = Domain(
    "company", "link", "company", (
        enum("industry", "industry", ("software", "logistics", "shipping", "brewing",
                                      "mining", "insurance", "banking", "pharmaceuticals",
                                      "textiles", "aerospace", "retail", "publishing",
                                      "telecoms"),
             syn=(("logistics", "shipping"), ("insurance", "banking"),
                  ("software", "telecoms"), ("retail", "publishing")),
             alias=(("software", "writes code for clients"),
                    ("logistics", "moves freight by road"),
                    ("shipping", "runs cargo vessels"), ("brewing", "makes beer"),
                    ("mining", "digs coal and ore"), ("insurance", "sells policies"),
                    ("banking", "takes deposits"), ("pharmaceuticals", "makes medicines"),
                    ("textiles", "weaves cloth"), ("aerospace", "builds aircraft parts"),
                    ("retail", "runs high-street shops"), ("publishing", "prints books"),
                    ("telecoms", "runs phone networks")),
             alias_clause=True, say="works in {x}", neg="does not work in {x}",
             ops=ENUM_OPS),
        enum("hq", "headquarters", ("Leeds", "Bristol", "Manchester", "Glasgow", "Dublin",
                                    "Rotterdam", "Oslo", "Lyon", "Porto", "Toronto",
                                    "Denver", "Perth"),
             say="is headquartered in {x}", neg="is not headquartered in {x}",
             ops=ENUM_OPS),
        num("founded", "founding year", tuple(range(1850, 2021, 5)), say="was founded {x}",
            scale="time", ops=TIME_OPS, decades=True),
        flag("listed", "listing", ", listed", "is publicly listed", "is not listed"),
    ),
    "{industry}, {hq}, est. {founded}{listed}",
    key="entity",
    names=("Meridian", "Apex", "Summit", "Northstar", "Keystone", "Pinnacle", "Horizon",
           "Vanguard", "Beacon", "Sterling"),
    intro=("An invoice from {name} turned up in the audit.",
           "{name} is quoted in the trade press this week.",
           "The contract names {name} as supplier.",
           "Analysts mentioned {name} on the earnings call."),
    sub="This {name}", cpt=2.55)

FILM = Domain(
    "film", "link", "film", (
        num("year", "release year", tuple(range(1930, 2025)), say="was released {x}",
            scale="time", ops=TIME_OPS, decades=True),
        enum("genre", "genre", ("drama", "thriller", "comedy", "horror", "western",
                                "documentary", "musical", "war film", "science fiction",
                                "romance", "crime drama", "psychological thriller"),
             syn=(("thriller", "psychological thriller"), ("drama", "crime drama"),
                  ("comedy", "romance"), ("western", "war film")),
             say="is {ax}", neg="is not {ax}", ops=ENUM_OPS),
        enum("country", "country", ("British", "American", "Canadian", "Australian",
                                    "French", "Irish", "Danish", "Korean", "Italian",
                                    "German"),
             syn=(("British", "Irish"), ("American", "Canadian"), ("Danish", "German")),
             alias=(("British", "was made in Britain"), ("American", "was made in the US"),
                    ("Canadian", "was made in Canada"),
                    ("Australian", "was made in Australia"),
                    ("French", "was made in France"), ("Irish", "was made in Ireland"),
                    ("Danish", "was made in Denmark"), ("Korean", "was made in Korea"),
                    ("Italian", "was made in Italy"), ("German", "was made in Germany")),
             alias_clause=True, say="is {x}", neg="is not {x}", ops=ENUM_OPS),
        num("runtime", "runtime", tuple(range(78, 186, 6)), fmt="{n} min", say="runs {x}",
            ops=QTY_OPS),
        flag("bw", "black and white", ", black-and-white", "was shot in black and white",
             "was not shot in black and white"),
    ),
    "{year} {country} {genre}, {runtime}{bw}",
    key="entity",
    names=("The Crossing", "Homecoming", "Blackwater", "The Long Road", "Undertow",
           "Northern Lights", "Crossfire", "The Visitor", "Aftermath", "Stranger"),
    intro=("A reviewer recommended {name} this weekend.",
           "The festival programme lists a screening of {name}.",
           "{name} came up in a quiz question.",
           "Someone asked the library for a copy of {name}."),
    sub="This {name}", cpt=2.7)

SHIP = Domain(
    "ship", "link", "ship", (
        enum("type", "type", ("frigate", "destroyer", "corvette", "ferry", "liner",
                              "trawler", "tanker", "schooner", "tug", "icebreaker",
                              "research vessel"),
             syn=(("frigate", "destroyer", "corvette"), ("ferry", "liner"),
                  ("trawler", "tug")),
             say="is {ax}", neg="is not {ax}", ops=ENUM_OPS),
        num("launched", "launch year", tuple(range(1850, 2021, 2)), say="was launched {x}",
            scale="time", ops=TIME_OPS, decades=True),
        enum("operator", "operator", ("Royal Navy", "Royal Canadian Navy", "US Navy",
                                      "Irish Naval Service", "Stena Line", "P&O",
                                      "Maersk", "NOAA", "British Antarctic Survey",
                                      "Cunard"),
             syn=(("Royal Navy", "Royal Canadian Navy"), ("Stena Line", "P&O")),
             say="served with {x}", neg="never served with {x}", ops=ENUM_OPS),
        enum("fate", "fate", ("in service", "scrapped", "wrecked", "preserved", "laid up",
                              "sold abroad"),
             syn=(("scrapped", "wrecked"), ("preserved", "laid up")),
             say="is now {x}", neg="is not {x}", ops=("eq", "ne")),
    ),
    "{type}, built {launched}, {operator}, {fate}",
    key="entity",
    names=("Vigilant", "Endeavour", "Resolute", "Intrepid", "Discovery", "Valiant",
           "Pioneer", "Enterprise"),
    intro=("A photograph in the museum is captioned {name}.",
           "The logbook records a meeting with the {name}.",
           "A model of the {name} was sold at auction.",
           "The harbour master's notes mention the {name}."),
    sub="This {name}", cpt=2.5)

RIVER = Domain(
    "river", "link", "river", (
        enum("country", "country", ("England", "Scotland", "Wales", "Ireland", "Australia",
                                    "Canada", "New Zealand", "Tasmania"),
             syn=(("England", "Scotland", "Wales"), ("Australia", "Tasmania")),
             alias=(("England", "is an English river"), ("Scotland", "is a Scottish river"),
                    ("Wales", "is a Welsh river"), ("Ireland", "is an Irish river"),
                    ("Australia", "is on the Australian mainland"),
                    ("Canada", "is a Canadian river"), ("New Zealand", "is a Kiwi river"),
                    ("Tasmania", "is a Tasmanian river")),
             alias_clause=True, say="flows through {x}", neg="does not flow through {x}",
             ops=ENUM_OPS),
        num("length", "length", (12, 18, 25, 32, 40, 52, 62, 75, 85, 110, 145, 190, 240,
                                 300, 380),
            fmt="{n} km", say="is {x} long", ops=QTY_OPS),
        enum("mouth", "mouth", ("North Sea", "Irish Sea", "Bristol Channel",
                                "English Channel", "Tasman Sea", "Pacific", "Atlantic",
                                "Firth of Forth", "Solway Firth", "Lake Ontario"),
             syn=(("North Sea", "Irish Sea"), ("Firth of Forth", "Solway Firth"),
                  ("Pacific", "Atlantic")),
             say="empties into the {x}", neg="does not empty into the {x}", ops=ENUM_OPS),
        num("basin", "catchment", (40, 90, 150, 240, 380, 520, 700, 950, 1300, 1800, 2500,
                                   3400, 4600),
            fmt="basin {n} km2", tshow=lambda n: f"{n} km2", say="drains {x}",
            ops=QTY_OPS),
        flag("navigable", "navigability", ", navigable", "is navigable",
             "is not navigable"),
    ),
    "{country}, {length}, {mouth}, {basin}{navigable}",
    key="entity",
    names=("Avon", "Esk", "Don", "Ouse", "Derwent", "Colne", "Wye", "Frome"),
    intro=("A walking guide describes the banks of the {name}.",
           "Flood warnings were issued for the {name}.",
           "An angling club fishes the {name}.",
           "A canoeist wrote about paddling the {name}."),
    sub="This {name}", cpt=2.4)

# -- nav -----------------------------------------------------------------------------------
DOCS_SITE = Domain(
    "docs_site", "nav", "documentation page", (
        enum("lang", "language", ("en", "de", "fr", "es", "ja", "pt", "it", "nl", "ko"),
             alias=(("en", "English"), ("de", "German"), ("fr", "French"),
                    ("es", "Spanish"), ("ja", "Japanese"), ("pt", "Portuguese"),
                    ("it", "Italian"), ("nl", "Dutch"), ("ko", "Korean")),
             say="in {x}", ops=("eq", "in")),
        enum("product", "product", ("cli", "sdk", "api", "console", "mobile", "terraform",
                                    "agent"),
             syn=(("cli", "sdk", "api"), ("console", "mobile")),
             say="for the {x}", neg="for any product but the {x}", ops=("eq",)),
        num("version", "version", (10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 50),
            show=_ver, say="for version {x}", scale="ver", ops=(">=", "<", "eq")),
        enum("section", "section", ("install", "quickstart", "tutorials", "reference",
                                    "changelog", "migration", "faq", "troubleshooting"),
             syn=(("quickstart", "tutorials", "install"), ("changelog", "migration"),
                  ("faq", "troubleshooting")),
             alias=(("install", "setup instructions"), ("quickstart", "getting-started guide"),
                    ("tutorials", "worked lessons"), ("reference", "API listing"),
                    ("changelog", "release history"), ("migration", "upgrade guide"),
                    ("faq", "common questions"), ("troubleshooting", "error fixes")),
             say="with the {x}", ops=("eq", "in")),
        enum("channel", "channel", ("stable", "beta", "nightly", "lts"),
             syn=(("stable", "lts"), ("beta", "nightly")),
             say="on the {x} channel", neg="on any channel but {x}", ops=("eq", "ne")),
    ),
    "{section}", key="url", url="/{lang}/{product}/{channel}/v{version}/{section}",
    intro=("Docs home", "Search results", "Release notes"), cpt=1.85)

SHOP = Domain(
    "shop", "nav", "product listing", (
        enum("category", "category", ("boots", "trainers", "sandals", "jackets", "coats",
                                      "jeans", "shirts", "dresses", "hats"),
             syn=(("boots", "trainers", "sandals"), ("jackets", "coats"),
                  ("shirts", "dresses")),
             say="for {x}", neg="for anything but {x}", ops=("eq",)),
        enum("brand", "brand", ("northfold", "kestrel", "alder", "moorland", "brightwater",
                                "tamsin", "holloway", "ravenglass"),
             say="from {x}", neg="from any brand but {x}", ops=ENUM_OPS),
        enum("colour", "colour", ("black", "white", "navy", "olive", "tan", "grey", "red",
                                  "burgundy", "cream", "khaki"),
             syn=(("black", "grey"), ("olive", "khaki", "tan"), ("red", "burgundy"),
                  ("white", "cream")),
             say="in {x}", ops=ENUM_OPS),
        enum("size", "size", ("xs", "s", "m", "l", "xl", "xxl"),
             alias=(("xs", "extra small"), ("s", "small"), ("m", "medium"),
                    ("l", "large"), ("xl", "extra large"), ("xxl", "double extra large")),
             say="in size {x}", ops=("eq", "in")),
        flag("sale", "sale", "?sale", "that are on sale", "that are not on sale"),
    ),
    "{brand} {category}", key="url", url="/{category}/{brand}/{colour}/{size}{sale}",
    intro=("Homepage", "Search results", "New arrivals"), cpt=2.3)

NEWS = Domain(
    "news_archive", "nav", "news article", (
        enum("region", "region", ("wales", "scotland", "london", "north-east", "midlands",
                                  "northern-ireland", "south-west"),
             alias=(("wales", "Cardiff"), ("scotland", "Edinburgh"), ("london", "Westminster"),
                    ("north-east", "Newcastle"), ("midlands", "Birmingham"),
                    ("northern-ireland", "Belfast"), ("south-west", "Plymouth")),
             say="from the {x} newsroom", ops=("eq", "in")),
        enum("section", "section", ("politics", "business", "sport", "science", "health",
                                    "technology", "culture"),
             syn=(("science", "technology", "health"), ("business", "politics")),
             say="in the {x} section", neg="in any section but {x}", ops=("eq", "in")),
        num("date", "date", _ym(2014, 2025), show=_ym_url, tshow=_ym_text,
            say="published {x}", scale="time", ops=(">=", "<=", ">", "<", "range")),
        enum("kind", "format", ("analysis", "explainer", "interview", "report", "live",
                                "obituary"),
             syn=(("analysis", "explainer"), ("report", "live")),
             say="that is {ax}", neg="that is not {ax}", ops=("eq", "ne")),
        enum("slug", "story", ("bus-strike", "harbour-plan", "school-closures",
                               "flood-defences", "wind-farm", "hospital-waits",
                               "rail-timetable", "council-budget", "stadium-bid",
                               "water-bills")),
    ),
    "", key="url", url="/{region}/{section}/{date}/{kind}-{slug}",
    intro=("Front page", "Most read", "Topic index"), cpt=2.35)

FORUM = Domain(
    "forum", "nav", "thread list", (
        enum("board", "board", ("hardware", "networking", "gaming", "linux", "windows",
                                "mac", "mobile", "security"),
             syn=(("linux", "windows", "mac"), ("hardware", "networking")),
             say="on the {x} board", neg="on any board but {x}", ops=("eq", "in")),
        enum("sort", "sort order", ("newest", "oldest", "top", "unanswered", "active"),
             syn=(("newest", "active"), ("top", "unanswered")),
             alias=(("newest", "most recent first"), ("oldest", "earliest first"),
                    ("top", "highest voted"), ("unanswered", "no replies yet"),
                    ("active", "latest activity")),
             say="sorted by {x}", ops=("eq",)),
        num("page", "page", tuple(range(1, 31)), say="on page {x}", ops=("eq", "range")),
        num("year", "year", tuple(range(2009, 2026)), say="posted {x}", scale="time",
            ops=("eq", ">=", "<")),
        flag("solved", "solved filter", "?solved", "showing only solved threads",
             "not limited to solved threads"),
    ),
    "{board} threads", key="url", url="/{board}/{year}/{sort}/p{page}{solved}",
    intro=("Forum index", "Search results", "Your subscriptions"), cpt=2.75)

GOV = Domain(
    "gov_portal", "nav", "form", (
        enum("dept", "department", ("tax", "transport", "health", "housing", "education",
                                    "benefits", "justice"),
             syn=(("tax", "benefits"), ("health", "housing")),
             say="from the {x} department", neg="from any department but {x}",
             ops=("eq",)),
        num("year", "year", tuple(range(2012, 2027)), say="dated {x}", scale="time",
            ops=("eq", ">=", "<")),
        enum("fmt", "file format", ("pdf", "docx", "odt", "html", "rtf"),
             alias=(("pdf", "an Acrobat file"), ("docx", "a Word file"),
                    ("odt", "an OpenDocument file"), ("html", "a web page"),
                    ("rtf", "a rich text file")),
             syn=(("docx", "odt", "rtf"),), say="as {x}", ops=("eq", "in")),
        enum("lang", "language", ("en", "cy", "ga", "gd"),
             alias=(("en", "English"), ("cy", "Welsh"), ("ga", "Irish"), ("gd", "Gaelic")),
             say="in {x}", ops=("eq",)),
        enum("form", "form", ("claim", "renewal", "appeal", "registration", "complaint",
                              "change-of-address", "refund")),
    ),
    "{form} form", key="url", url="/{dept}/{year}/{form}.{fmt}?{lang}",
    intro=("Services A-Z", "Search results", "Popular forms"), cpt=2.5)

# -- route ---------------------------------------------------------------------------------
IT_HELPDESK = Domain(
    "it_helpdesk", "route", "queue", (
        enum("system", "system", ("email", "VPN", "laptops", "printers", "payroll", "CRM",
                                  "Wi-Fi", "SSO"),
             alias=(("email", "Outlook"), ("VPN", "the remote access client"),
                    ("laptops", "my notebook computer"), ("printers", "the copier"),
                    ("payroll", "the payslip system"), ("CRM", "the customer database"),
                    ("Wi-Fi", "the wireless network"), ("SSO", "single sign-on")),
             syn=(("VPN", "Wi-Fi"), ("email", "SSO")),
             say="The problem is with {x}.", ops=("eq",)),
        enum("issue", "issue", ("outage", "access request", "hardware fault",
                                "password reset", "slowness", "data loss"),
             alias=(("outage", "it is down for everyone"),
                    ("access request", "I need to be given permission"),
                    ("hardware fault", "something is physically broken"),
                    ("password reset", "I am locked out"),
                    ("slowness", "it is painfully sluggish"),
                    ("data loss", "files have disappeared")),
             syn=(("outage", "slowness"), ("access request", "password reset")),
             say="In short: {x}.", ops=("eq",)),
        enum("user", "requester", ("staff", "contractor", "executive", "intern"),
             alias=(("contractor", "an external consultant"),
                    ("executive", "a board member"),
                    ("intern", "on a summer placement"),
                    ("staff", "a permanent employee")),
             say="Requester: {x}.", ops=("eq",)),
        enum("site", "site", ("Leeds", "Cardiff", "Glasgow", "Belfast", "London", "Remote"),
             say="Site: {x}.", neg="Site: not {x}.", ops=("eq",)),
        flag("vip", "VIP", ", VIP", "I am on the VIP support list.",
             "I am not on the VIP list."),
    ),
    "{issue}, {system}, {user}, {site}{vip}", key="queue",
    intro=("IT ticket", "Service desk request", "Helpdesk chat"), cpt=2.8)

BANK = Domain(
    "bank_support", "route", "queue", (
        enum("product", "product", ("current account", "savings", "mortgage",
                                    "credit card", "business loan", "ISA"),
             alias=(("current account", "my everyday account"),
                    ("savings", "my rainy-day pot"), ("mortgage", "my home loan"),
                    ("credit card", "my plastic"), ("business loan", "my company's borrowing"),
                    ("ISA", "my tax-free wrapper")),
             syn=(("current account", "savings", "ISA"), ("mortgage", "business loan")),
             say="It concerns {x}.", ops=("eq",)),
        enum("issue", "issue", ("fraud", "dispute", "failed payment", "statement",
                                "closure", "overdraft"),
             alias=(("fraud", "a payment I never made"),
                    ("dispute", "a merchant who will not refund me"),
                    ("failed payment", "a transfer that bounced"),
                    ("statement", "a missing monthly summary"),
                    ("closure", "shutting it down"),
                    ("overdraft", "going below zero")),
             syn=(("fraud", "dispute"), ("statement", "closure")),
             say="The issue is {x}.", ops=("eq",)),
        enum("segment", "segment", ("retail", "premier", "business", "private"),
             syn=(("premier", "private"),),
             say="I am a {x} customer.", neg="I am not a {x} customer.", ops=("eq",)),
        enum("lang", "language", ("English", "Welsh", "Polish", "Urdu", "Punjabi"),
             say="Please reply in {x}.", neg="Any language but {x} is fine.", ops=("eq",)),
        flag("urgent", "urgency", ", urgent", "This is urgent.", "This is not urgent."),
    ),
    "{issue}, {product}, {segment}, {lang}{urgent}", key="queue",
    intro=("Secure message", "Web chat", "Email to support"), cpt=3.0)

TELECOM = Domain(
    "telecom_support", "route", "queue", (
        enum("service", "service", ("broadband", "fibre", "mobile", "landline", "TV"),
             alias=(("broadband", "my home internet"), ("fibre", "my FTTP line"),
                    ("mobile", "my phone contract"), ("landline", "my home phone"),
                    ("TV", "my set-top box")),
             syn=(("broadband", "fibre"), ("mobile", "landline")),
             say="The affected service is {x}.", ops=("eq",)),
        enum("issue", "issue", ("no signal", "billing error", "slow speed", "number porting",
                                "roaming", "faulty device"),
             alias=(("no signal", "nothing connects at all"),
                    ("billing error", "I was charged twice"),
                    ("slow speed", "it crawls in the evenings"),
                    ("number porting", "I want to keep my old number"),
                    ("roaming", "it stopped working abroad"),
                    ("faulty device", "the router keeps rebooting")),
             syn=(("no signal", "slow speed"), ("roaming", "number porting")),
             say="Problem: {x}.", ops=("eq",)),
        enum("plan", "plan", ("pay monthly", "SIM only", "pay as you go", "business"),
             syn=(("pay monthly", "SIM only", "pay as you go"),),
             say="I am on a {x} plan.", neg="I am not on a {x} plan.", ops=("eq",)),
        enum("region", "region", ("North", "South", "Midlands", "Scotland", "Wales",
                                  "Northern Ireland"),
             say="Region: {x}.", neg="Region: not {x}.", ops=("eq",)),
        flag("priority", "priority care", ", priority care",
             "I am registered for priority care.", "I am not registered for priority care."),
    ),
    "{service} {issue}, {plan}, {region}{priority}", key="queue",
    intro=("Support ticket", "Live chat", "Complaint form"), cpt=3.0)

HOSPITAL = Domain(
    "hospital_admin", "route", "queue", (
        enum("dept", "department", ("cardiology", "radiology", "orthopaedics",
                                    "paediatrics", "oncology", "dermatology"),
             alias=(("cardiology", "my heart clinic"), ("radiology", "my scan"),
                    ("orthopaedics", "my knee operation"),
                    ("paediatrics", "my child's care"),
                    ("oncology", "my cancer treatment"), ("dermatology", "my skin clinic")),
             syn=(("cardiology", "radiology"), ("oncology", "dermatology")),
             say="This is about {x}.", ops=("eq",)),
        enum("request", "request", ("appointment change", "records request", "referral",
                                    "billing query", "transport"),
             alias=(("appointment change", "I need to move my slot"),
                    ("records request", "I would like a copy of my notes"),
                    ("referral", "I need a letter to see a specialist"),
                    ("billing query", "there is a charge I do not understand"),
                    ("transport", "I need a lift to the hospital")),
             syn=(("appointment change", "referral"),),
             say="Request: {x}.", ops=("eq",)),
        enum("patient", "patient type", ("inpatient", "outpatient", "day case"),
             syn=(("inpatient", "outpatient", "day case"),),
             say="I am {ax}.", neg="I am not {ax}.", ops=("eq",)),
        enum("site", "site", ("General", "St Mary's", "Riverside", "Northgate"),
             say="Site: {x}.", neg="Site: not {x}.", ops=("eq",)),
        flag("interpreter", "interpreter", ", interpreter", "I need an interpreter.",
             "I do not need an interpreter."),
    ),
    "{request}, {dept}, {patient}, {site}{interpreter}", key="queue",
    intro=("Patient enquiry", "Switchboard note", "Online form"), cpt=3.2)

UNIVERSITY = Domain(
    "university_services", "route", "queue", (
        enum("faculty", "faculty", ("engineering", "law", "medicine", "arts", "science",
                                    "business", "education", "nursing", "music"),
             alias=(("engineering", "the civil and mechanical school"),
                    ("law", "the legal studies school"), ("medicine", "the medical school"),
                    ("arts", "the humanities"), ("science", "the natural sciences"),
                    ("business", "the MBA programme"),
                    ("education", "the teacher training school"),
                    ("nursing", "the school of midwifery and care"),
                    ("music", "the conservatoire")),
             syn=(("engineering", "science"), ("arts", "law")),
             say="I study in {x}.", ops=("eq",)),
        enum("request", "request", ("fees", "enrolment", "accommodation", "exam board",
                                    "visa letter", "transcript", "library fine",
                                    "timetable clash", "graduation"),
             alias=(("fees", "my tuition payment"), ("enrolment", "registering for the year"),
                    ("accommodation", "my room in halls"),
                    ("exam board", "a resit decision"),
                    ("visa letter", "a letter for immigration"),
                    ("transcript", "an official record of my marks"),
                    ("library fine", "an overdue book charge"),
                    ("timetable clash", "two lectures at the same hour"),
                    ("graduation", "my degree ceremony")),
             syn=(("enrolment", "transcript"),),
             say="I need help with {x}.", ops=("eq",)),
        enum("level", "level", ("undergraduate", "taught postgraduate",
                                "research postgraduate", "exchange"),
             syn=(("taught postgraduate", "research postgraduate"),),
             say="I am {ax} student.", neg="I am not {ax} student.", ops=("eq",)),
        enum("campus", "campus", ("City", "Park", "Harbour", "Medical"),
             say="Campus: {x}.", neg="Campus: not {x}.", ops=("eq",)),
        flag("intl", "international", ", international", "I am an international student.",
             "I am not an international student."),
    ),
    "{request}, {faculty}, {level}, {campus}{intl}", key="queue",
    intro=("Student enquiry", "Portal message", "Front desk note"), cpt=3.35)


# family -> (trained, dev-held-out, test-held-out)
DOMAINS = {
    "match": ((BACKPACKS, FASTENERS, LAPTOPS, API_ENDPOINTS), (TYRES,), (CLOUD_PLANS,)),
    "link": ((PERSON, SETTLEMENT, COMPANY, FILM), (SHIP,), (RIVER,)),
    "nav": ((DOCS_SITE, SHOP, NEWS), (FORUM,), (GOV,)),
    "route": ((IT_HELPDESK, BANK, TELECOM), (HOSPITAL,), (UNIVERSITY,)),
}
FAMILIES = tuple(DOMAINS)
TRAINED = {f: tuple(d.name for d in v[0]) for f, v in DOMAINS.items()}
DEV_HELDOUT = {f: tuple(d.name for d in v[1]) for f, v in DOMAINS.items()}
TEST_HELDOUT = {f: tuple(d.name for d in v[2]) for f, v in DOMAINS.items()}
ALL_DOMAINS = {d.name: d for v in DOMAINS.values() for grp in v for d in grp}


# ---- values ------------------------------------------------------------------------------

def pool(a: Attr) -> list:
    if a.kind == "num":
        out = [(n, 0) for n in a.values]
        for i, (_, _, nums, _) in enumerate(a.alt, 1):
            out += [(n, i) for n in nums]
        return out
    return list(a.values)


def metric(a: Attr, v) -> float:
    n, u = v
    return float(n) if u == 0 else n * a.alt[u - 1][1]


def fmt_num(n) -> str:
    if isinstance(n, float):
        return f"{n:g}"
    return f"{n:,}" if n >= 10000 else str(n)


def show(a: Attr, v) -> str:
    """The value as an option prints it."""
    if a.kind == "num":
        n, u = v
        if u:
            return a.alt[u - 1][3].format(n=fmt_num(n))
        if a.show:
            return a.show(n)
        return a.fmt.format(n=fmt_num(n))
    if a.kind == "flag":
        return a.text if v else ""
    return str(v)


def tshow(a: Attr, n) -> str:
    """A base-unit number as the state states it."""
    if a.tshow:
        return a.tshow(n)
    return show(a, (n, 0))


def aliased(a: Attr) -> bool:
    """Every value can be named without the option's words."""
    keys = {k for k, _ in a.alias}
    return a.kind == "enum" and all(v in keys for v in a.values)


def alias_of(a: Attr, v) -> str:
    return next(w for k, w in a.alias if k == v)


def art(w: str) -> str:
    return ("an " if w[:1].lower() in "aeiou" else "a ") + w


def syn_mates(a: Attr, v) -> list:
    out: list = []
    for g in a.syn:
        if v in g:
            out += [x for x in g if x != v and x not in out]
    return out


# ---- constraints -------------------------------------------------------------------------

@dataclass(eq=False)
class Con:
    attr: Attr
    op: str
    t: Any = None                    # num: threshold (base metric); enum eq/ne: the value
    hi: Any = None                   # num range: upper bound
    vals: tuple = ()                 # enum in: the two values
    # how the state names the axis (see the module docstring): "gold" names the gold's
    # value, "fail" one failing value, "none" nothing, "all" every value on the axis
    case: str = "none"
    literal: bool = True             # the requirement is stated in the option's words
    traps: list = field(default_factory=list)   # failing values mentioned in passing

    def ok(self, v) -> bool:
        a, op = self.attr, self.op
        if a.kind == "flag":
            return bool(v) == (op == "need")
        if a.kind == "enum":
            if op == "eq":
                return v == self.t
            if op == "ne":
                return v != self.t
            return v in self.vals
        m = metric(a, v)
        eps = 1e-9
        if op == ">=":
            return m >= self.t - eps
        if op == "<=":
            return m <= self.t + eps
        if op == ">":
            return m > self.t + eps
        if op == "<":
            return m < self.t - eps
        if op == "eq":
            return abs(m - self.t) < 1e-6
        return self.t - eps <= m <= self.hi + eps          # range, inclusive

    def numbers(self) -> set:
        """The numbers the statement prints (num constraints)."""
        if self.attr.kind != "num":
            return set()
        return {self.t, self.hi} - {None}


def capacity(a: Attr, op: str) -> int:
    """The most values one grid axis can carry under this op: one satisfying, the rest not."""
    if a.kind == "flag" or op == "ne":
        return 2
    n = len(pool(a))
    if a.kind == "enum" and not aliased(a):
        n = min(n, NOALIAS_CAP + (1 if op == "in" else 0))
    if op == "in":
        return n - 1
    return n if op == "eq" else n - 1


def draw_case(a: Attr, op: str, s: int, rng: random.Random) -> str:
    """Which of the axis's s values the state will name. Exchangeable: P(the gold's value
    is named) equals P(any given failing value is named)."""
    if a.kind == "flag":
        return "gold" if op == "need" else "fail"      # the qualifier's words either way
    if a.kind == "enum" and not aliased(a):
        if s == 2:
            return rng.choice(("gold", "fail", "all"))
        return "all"
    r = rng.random()
    return "gold" if r < MU / s else "fail" if r < MU else "none"


def fit_op(a: Attr, op: str, case: str, rng: random.Random) -> str:
    """The op the case can be stated with."""
    if a.kind == "flag":
        return op
    if a.kind == "enum":
        if case == "fail" and not aliased(a):
            return "ne"                                  # "anything but X", X on the axis
        return op if op in ("eq", "in") else "eq"
    if case == "gold":                                   # the threshold IS the gold
        return {">": ">=", "<": "<="}.get(op, op)
    if op == "eq":                                       # "exactly 16 GB" names the gold
        # drawn, not the first listed: always ">=" made the gold the larger number of
        # a hard-negative pair 0.69 of the time on vCPUs
        return rng.choice([o for o in a.ops if o != "eq"] or ["range"])
    return op


def _decade_range(a: Attr, rng: random.Random) -> tuple[int, int] | None:
    lo, hi = min(a.values), max(a.values)
    d = rng.randrange(lo // 10 * 10, hi // 10 * 10 + 1, 10)
    part = rng.choice(((0, 3), (4, 6), (7, 9), (0, 9)))
    a0, b0 = d + part[0], d + part[1]
    if a0 < lo or b0 > hi:
        return None
    return a0, b0


def make_con(a: Attr, op: str, size: int, case: str,
             rng: random.Random) -> tuple[Con, list, list] | None:
    """A constraint on `a` with at least `size - 1` failing values. -> (con, sat, fail)."""
    vals = pool(a)
    for _ in range(40):
        if a.kind == "flag":
            con = Con(a, op)
        elif a.kind == "enum":
            if op == "in":
                con = Con(a, op, vals=tuple(rng.sample(vals, 2)))
            else:
                con = Con(a, op, t=rng.choice(vals))
        else:
            base = sorted({n for n in a.values})
            if op == "range":
                span = None
                if a.decades and case != "gold" and rng.random() < 0.6:
                    span = _decade_range(a, rng)
                if span is None:
                    i = rng.randrange(len(base) - 1)
                    j = min(len(base) - 1, i + rng.randint(1, 3))
                    span = (base[i], base[j])
                con = Con(a, op, t=span[0], hi=span[1])
            else:
                con = Con(a, op, t=rng.choice(base))
        con.case = case
        sat = [v for v in vals if con.ok(v)]
        fail = [v for v in vals if not con.ok(v)]
        if a.kind == "num" and case != "gold":
            # nothing on the axis prints a number the statement prints, bar the boundary
            # entry a strict threshold is stated against
            nums = con.numbers()
            sat = [v for v in sat if v[0] not in nums]
            fail = [v for v in fail if v[0] not in nums or
                    (case == "fail" and op in (">", "<") and v == (con.t, 0))]
        if a.kind == "num" and case == "gold" and (con.t, 0) not in sat and \
                (con.hi, 0) not in sat:
            continue
        if sat and len(fail) >= size - 1:
            return con, sat, fail
    return None


def _distance(con: Con, v) -> float:
    a = con.attr
    m = metric(a, v)
    if con.op == "range":
        return max(con.t - m, m - con.hi, 0.0)
    return abs(m - con.t)


def gold_axis(con: Con, sat: list, fail: list, size: int, rng: random.Random) -> list:
    """The gold block's values on one axis: the gold's first, then `size - 1` failing ones,
    drawn near the constraint so the one-edit neighbours are *minimal* edits. Sets the
    statement's literalness and its passing mentions from `con.case`."""
    a, case = con.attr, con.case
    if a.kind == "flag":
        con.literal = True
        return [con.op == "need", con.op != "need"]
    if a.kind == "enum":
        if con.op == "ne":
            mates = [v for v in syn_mates(a, con.t) if v in sat]
            g = rng.choice(mates) if mates and rng.random() < 0.6 else rng.choice(sat)
            axis = [g, con.t]
        else:
            g = con.t if con.op == "eq" else rng.choice(con.vals)
            first = [v for v in syn_mates(a, g) if v in fail]
            rng.shuffle(first)
            rest = [v for v in fail if v not in first]
            rng.shuffle(rest)
            axis = [g] + (first + rest)[: size - 1]
        con.literal = case in ("gold", "all") or con.op == "ne"
        if case == "all":
            con.traps = axis[1:]
        elif case == "fail" and con.op != "ne":
            con.traps = [rng.choice(axis[1:])]
        return axis
    # num
    if case == "gold":
        ends = [(x, 0) for x in (con.t, con.hi) if x is not None and (x, 0) in sat]
        g = rng.choice(ends)
        fail = [v for v in fail if v[0] != g[0]]        # only the gold prints its number
    else:
        g = rng.choice(sat)          # any unit: the state's unit word must not mark the gold
    picked: list = []
    boundary = case == "fail" and con.op in (">", "<") and (con.t, 0) in fail
    if boundary:
        picked.append((con.t, 0))                        # "over 30 L" beside a 30 L entry
    if a.alt and size > 1 and rng.random() < P_UNIT_FLIP:
        flips = [v for v in fail if v[0] == g[0] and v[1] != g[1] and v not in picked]
        if flips and len(picked) < size - 1:
            picked.append(rng.choice(flips))
    near = sorted((v for v in fail if v not in picked), key=lambda v: _distance(con, v))
    window = near[: max(2 * (size - 1) + 2, size - 1)]
    rng.shuffle(window)
    picked += window[: max(0, size - 1 - len(picked))]
    axis = [g] + picked
    con.literal = case == "gold"
    if case == "fail" and not boundary and len(axis) > 1:
        con.traps = [rng.choice(axis[1:])]
    return axis


def dims_for(block: int, caps: list[int], rng: random.Random) -> list[int]:
    """Axis sizes whose product first reaches `block`, grown in proportion to a random
    weight per axis so grid shapes vary from square to skewed."""
    a = [min(2, c) for c in caps]
    w = [rng.uniform(0.5, 2.0) for _ in caps]
    while math.prod(a) < block:
        cands = [j for j in range(len(caps)) if a[j] < caps[j]]
        if not cands:
            break
        j = max(cands, key=lambda j: (w[j] / a[j], rng.random()))
        a[j] += 1
    return a


# ---- the instance ------------------------------------------------------------------------

@dataclass(eq=False)
class Inst:
    dom: Domain
    cons: list
    cells: list                      # option order: {attr name: value}
    block_of: list                   # option -> block index
    gold: int
    hn: list                         # option indices one edit from the gold
    hn_kind: list
    hn_attr: list
    shape: list                      # the gold block's grid shape
    blocks: int
    axes: dict = field(default_factory=dict)         # attr name -> gold block's values

    @property
    def n(self) -> int:
        return len(self.cells)

    def fails(self, i: int) -> list[str]:
        return [c.attr.name for c in self.cons if not c.ok(self.cells[i][c.attr.name])]


def draw_n(rng: random.Random, lo: int = N_MIN, hi: int = N_MAX) -> int:
    """Log-uniform over [lo, hi]: as many draws in 2-10 as in 10-50, a tail to 512."""
    return min(hi, int(math.exp(rng.uniform(math.log(lo), math.log(hi + 1)))))


def _weighted(rng: random.Random, pairs) -> Any:
    r = rng.random() * sum(w for _, w in pairs)
    for v, w in pairs:
        r -= w
        if r <= 0:
            return v
    return pairs[-1][0]


def _block_sizes(n: int, rng: random.Random, max_blocks: int) -> list[int]:
    """How the option set splits into blocks (entries sharing their unconstrained values):
    one block about half the time, otherwise 2-5 at random cut points, never more than
    there are distinct unconstrained tuples."""
    if n < 2 or max_blocks < 2 or rng.random() < P_ONE_BLOCK:
        return [n]
    m = min(rng.randint(2, MAX_BLOCKS), max_blocks, n)
    cuts = sorted(rng.sample(range(1, n), m - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [n])]


def _edit_kind(a: Attr, g, v) -> str:
    if a.kind == "flag":
        return "drop" if g else "add"
    if a.kind == "enum":
        return "synonym" if v in syn_mates(a, g) else "value"
    return "unit" if g[0] == v[0] and g[1] != v[1] else "value"


def build(dom: Domain, rng: random.Random, n: int, large: bool = False) -> Inst | None:
    """One option set of (up to) `n` entries with exactly one satisfying every constraint.

    The entries are a uniform random N-subset of the grid over the constrained axes, dealt
    at random into blocks that share their unconstrained values; the gold is then a
    uniformly random entry, and only after it is drawn are its coordinates mapped to the
    satisfying values (the other coordinates to the failing ones, in random order). So the
    gold is exchangeable with every other entry in everything a reader could count --
    how often its values recur, how big its block is, how many one-edit neighbours it
    has. The second version fixed the gold's values first and grew grids around them: each
    block was drawn uniformly (a one-entry block held the gold far above 1/N) and every
    other block left the gold's value off an axis, so "the entry whose words are rarest"
    scored 0.20-0.32 against a 0.10 chance.
    """
    cand = [a for a in dom.attrs if a.ops]
    flags = [a for a in cand if a.kind == "flag"]
    others = [a for a in cand if a.kind != "flag"]
    k = _weighted(rng, K_WEIGHTS_LARGE if large else K_WEIGHTS)
    # Every entry is a distinct combination of the constrained values, so a large N needs
    # a grid that large: add constrained attributes until it fits (the last one, if need
    # be, leaves no unconstrained attribute and so one block).
    spare = 0 if len(dom.attrs) > len(cand) else 1
    for _ in range(12):
        kk = min(k, len(cand) - spare)
        chosen = rng.sample(others, min(kk, len(others)))
        if flags and len(chosen) >= 2 and (rng.random() < 0.35 or len(chosen) < kk):
            chosen[-1 if len(chosen) >= kk else len(chosen):] = [rng.choice(flags)]
        ops = [rng.choice(a.ops) for a in chosen]
        caps = [capacity(a, op) for a, op in zip(chosen, ops)]
        if math.prod(caps) >= n:
            break
        if kk < len(cand) - spare:
            k += 1
        elif spare:
            spare = 0
    k = len(chosen)
    free = [a for a in dom.attrs if a not in chosen]
    names = [a.name for a in chosen]
    shape = dims_for(n, caps, rng)
    cons, axes = [], []
    for j, (a, op) in enumerate(zip(chosen, ops)):
        # the case is drawn once: redrawing it when an axis will not build favoured the
        # cases that always build ("gold": 0.51 of tyre profiles against 0.3 drawn)
        case = draw_case(a, op, shape[j], rng)
        got = None
        while got is None and shape[j] >= 2:
            got = make_con(a, fit_op(a, op, case, rng), shape[j], case, rng)
            if got is None:
                shape[j] -= 1
        if got is None:
            return None
        con, sat, fail = got
        cons.append(con)
        axes.append(gold_axis(con, sat, fail, shape[j], rng))
    shape = [len(ax) for ax in axes]
    n = min(n, math.prod(shape))

    # blocks: one distinct unconstrained tuple each
    n_free = math.prod(len(pool(a)) for a in free) if free else 1
    sizes = _block_sizes(n, rng, n_free)
    seen: set = set()
    free_vals: list[dict] = []
    while len(free_vals) < len(sizes):
        fv = {a.name: rng.choice(pool(a)) for a in free}
        key = tuple(sorted((k2, repr(v)) for k2, v in fv.items()))
        if key not in seen:
            seen.add(key)
            free_vals.append(fv)

    # entries: n distinct grid coordinates, dealt into blocks, then a uniform gold
    size_g = math.prod(shape)
    if size_g <= 20000:
        coords = rng.sample(list(itertools.product(*[range(s_) for s_ in shape])), n)
    else:
        got_set: set = set()
        while len(got_set) < n:
            got_set.add(tuple(rng.randrange(s_) for s_ in shape))
        coords = list(got_set)
        rng.shuffle(coords)
    block_of = [b for b, size in enumerate(sizes) for _ in range(size)]
    gold = rng.randrange(n)
    maps = []
    for j, s_ in enumerate(shape):
        rest = axes[j][1:]
        rng.shuffle(rest)
        it = iter(rest)
        maps.append([axes[j][0] if idx == coords[gold][j] else next(it) for idx in range(s_)])
    cells = [{**free_vals[block_of[i]], **{names[j]: maps[j][c[j]] for j in range(k)}}
             for i, c in enumerate(coords)]
    order = list(range(n))
    rng.shuffle(order)
    cells = [cells[i] for i in order]
    block_of = [block_of[i] for i in order]
    gold = order.index(gold)

    # THE INVARIANT, checked on every entry rather than trusted from the construction
    sat = [i for i, c in enumerate(cells) if all(con.ok(c[con.attr.name]) for con in cons)]
    assert sat == [gold], f"{dom.name}: {len(sat)} entries satisfy every constraint"
    g = cells[gold]
    hn, hn_kind, hn_attr = [], [], []
    for i, c in enumerate(cells):
        if i == gold:
            continue
        diff = [a for a in dom.attrs if c[a.name] != g[a.name]]
        if len(diff) == 1 and diff[0] in chosen:
            hn.append(i)
            hn_kind.append(_edit_kind(diff[0], g[diff[0].name], c[diff[0].name]))
            hn_attr.append(diff[0].name)
    return Inst(dom, cons, cells, block_of, gold, hn, hn_kind, hn_attr, shape, len(sizes),
                axes={a.name: ax for a, ax in zip(chosen, axes)})


# ---- rendering ---------------------------------------------------------------------------

QTY = {">=": ("at least {t}", "{t} or more", "no less than {t}", "a minimum of {t}"),
       "<=": ("at most {t}", "{t} or less", "no more than {t}", "a maximum of {t}"),
       ">": ("more than {t}", "over {t}", "above {t}"),
       "<": ("less than {t}", "under {t}", "below {t}"),
       "range": ("between {a} and {b}", "{a} to {b}"),
       "eq": ("exactly {t}", "{t}")}
TIME = {">=": ("in {t} or later", "no earlier than {t}"),
        "<=": ("in {t} or earlier", "no later than {t}"),
        ">": ("after {t}", "later than {t}"),
        "<": ("before {t}", "earlier than {t}"),
        "range": ("between {a} and {b}",),
        "eq": ("in {t}",)}
VER = {">=": ("{t} or newer", "{t} or later"),
       "<": ("older than {t}", "before {t}"),
       "<=": ("{t} or older",),
       "eq": ("{t}", "exactly {t}")}
SYM = {">=": ">= {t}", "<=": "<= {t}", ">": "> {t}", "<": "< {t}", "range": "{a}-{b}",
       "eq": "{t}"}
_PARTS = {(0, 3): "the early {d}s", (4, 6): "the mid {d}s", (7, 9): "the late {d}s",
          (0, 9): "the {d}s"}


def num_phrase(con: Con, rng: random.Random, symbolic: bool = False) -> str:
    a = con.attr
    t = tshow(a, con.t) if con.t is not None else ""
    lo, hi = (tshow(a, con.t), tshow(a, con.hi)) if con.op == "range" else ("", "")
    if symbolic:
        return SYM[con.op].format(t=t, a=lo, b=hi)
    if a.scale == "time" and con.op == "range" and a.decades:
        part = (con.t % 10, con.hi % 10)
        if part in _PARTS and con.t // 10 == con.hi // 10 and not a.tshow:
            return ("in " if rng.random() < 0.5 else "during ") + \
                _PARTS[part].format(d=con.t // 10 * 10)
    table = {"time": TIME, "ver": VER}.get(a.scale, QTY)
    if con.op not in table:
        table = QTY
    return rng.choice(table[con.op]).format(t=t, a=lo, b=hi)


def fill(template: str, x: str) -> str:
    return template.replace("{ax}", art(x)).replace("{x}", x)


def _named(a: Attr, v, literal: bool) -> str:
    return str(v) if literal else alias_of(a, v)


def clause(con: Con, rng: random.Random) -> str:
    """The constraint as a fragment of prose, e.g. "with a capacity of at least 30 L"."""
    a = con.attr
    if a.kind == "flag":
        return a.need if con.op == "need" else a.avoid
    if a.kind == "num":
        return fill(a.say, num_phrase(con, rng))
    lit = con.literal
    if con.op == "eq":
        if not lit and a.alias_clause:
            return alias_of(a, con.t)
        return fill(a.say, _named(a, con.t, lit))
    if con.op == "in":
        if not lit and a.alias_clause:
            return " or ".join(alias_of(a, v) for v in con.vals)
        w1, w2 = (_named(a, v, lit) for v in con.vals)
        if "{ax}" in a.say:                  # "is a poet or a novelist"
            return a.say.replace("{ax}", f"{art(w1)} or {art(w2)}")
        either = rng.random() >= 0.6 and "the {x}" not in a.say     # not "the either X"
        return fill(a.say, f"either {w1} or {w2}" if either else f"{w1} or {w2}")
    w = str(con.t)
    if a.neg:
        return fill(a.neg, w)
    but = rng.choice(("anything but ", "anything except "))
    if "{ax}" in a.say:
        return a.say.replace("{ax}", but + art(w))
    return fill(a.say, but + w)


def spec_value(con: Con, rng: random.Random, symbolic: bool = False) -> str:
    """The constraint as a spec-sheet value, e.g. ">= 30 L" or "not black"."""
    a = con.attr
    if a.kind == "flag":
        return rng.choice(("required", "must have")) if con.op == "need" else \
            rng.choice(("not wanted", "must not have"))
    if a.kind == "num":
        return num_phrase(con, rng, symbolic)
    if con.op == "eq":
        return _named(a, con.t, con.literal)
    if con.op == "in":
        return " or ".join(_named(a, v, con.literal) for v in con.vals)
    return rng.choice(("not ", "anything except ", "any but ")) + str(con.t)


def trap_texts(con: Con) -> list[str]:
    """The failing values this constraint mentions in passing, as the options print them."""
    a = con.attr
    out = []
    for v in con.traps:
        if a.kind == "num":
            out.append(show(a, v) if v[1] else tshow(a, v[0]))
        else:
            out.append(str(v))
    return out


def join(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


# -- per family ----------------------------------------------------------------------------
_GREET = ("Hi,", "Hello,", "Hi there,", "Good morning,", "Hello team,")
_SIGN = ("Thanks!", "Many thanks.", "Cheers.", "Thank you.", "Best regards.")
_MATCH_TRAP = ("My current one is {x}.", "The last one we bought was {x}, which did not "
               "work out.", "A colleague suggested {x}, but that will not do.",
               "We already have one that is {x}.", "Last time we were sent {x} by mistake.")
_SPEC_TRAP = ("- Note: previous supply was {x}", "- Rejected earlier: {x}",
              "- Current stock: {x}")
_PO_TRAP = ("prev. {x}", "not as last order ({x})")
_LINK_TRAP = ("Not to be confused with the namesake who {c}.",
              "A different {name}, who {c}, is often mixed up with this one.",
              "(The {name} who {c} is someone else.)")
_NAV_TRAP = ("Last time you opened the '{x}' page by mistake.",
             "Not the '{x}' one.", "The '{x}' link is not it.")
_ROUTE_TRAP = ("An earlier ticket of mine mentioned {x}; another team handled it.",
               "(Last month's ticket said {x}; that one is sorted.)",
               "My colleague's ticket said {x}; mine is different.")


def _traps(cons: list, rng: random.Random, templates) -> list[str]:
    return [rng.choice(templates).format(x=x) for c in cons for x in trap_texts(c)]


def state_match(inst: Inst, rng: random.Random) -> tuple[str, str]:
    dom = inst.dom
    cons = inst.cons[:]
    rng.shuffle(cons)
    style = rng.choice(("message", "spec", "po"))
    if style == "message":
        body = f"I'm looking for {art(dom.noun)} {join([clause(c, rng) for c in cons])}."
        parts = [body] + _traps(cons, rng, _MATCH_TRAP)
        rng.shuffle(parts)
        return style, f"{rng.choice(_GREET)}\n\n{' '.join(parts)}\n\n{rng.choice(_SIGN)}"
    if style == "spec":
        lines = [f"- {c.attr.label.capitalize()}: {spec_value(c, rng)}" for c in cons]
        lines += _traps(cons, rng, _SPEC_TRAP)
        rng.shuffle(lines)
        head = f"Requirement R-{rng.randrange(100, 999)}: {dom.noun}"
        return style, head + "\n" + "\n".join(lines)
    items = [f"{c.attr.label} {spec_value(c, rng, symbolic=True)}" for c in cons]
    items += _traps(cons, rng, _PO_TRAP)
    rng.shuffle(items)
    return style, (f"PO-{rng.randrange(10000, 99999)} line {rng.randint(1, 9)}: "
                   f"{dom.noun}; {'; '.join(items)}; qty {rng.randint(1, 60)}")


def _say_value(con: Con, x: str) -> str:
    a = con.attr
    if a.kind == "num":
        x = ("in " + x) if a.scale == "time" else x
    return fill(a.say, x)


def state_link(inst: Inst, rng: random.Random, name: str) -> tuple[str, str]:
    dom = inst.dom
    cons = inst.cons[:]
    rng.shuffle(cons)
    style = rng.choice(("news", "note"))
    sub = dom.sub.format(name=name)
    intro = rng.choice(dom.intro).format(name=name)
    traps = [rng.choice(_LINK_TRAP).format(c=_say_value(c, x), name=name)
             for c in cons for x in trap_texts(c)]
    if style == "news":
        parts = [f"{sub} {join([clause(c, rng) for c in cons])}."] + traps
        rng.shuffle(parts)
        return style, intro + " " + " ".join(parts)
    lines = [f"{sub} {clause(c, rng)}." for c in cons] + traps
    rng.shuffle(lines)
    return style, f"Mention: \"{name}\"\n{intro}\n" + "\n".join(lines)


def state_nav(inst: Inst, rng: random.Random, here: str) -> tuple[str, str]:
    dom = inst.dom
    cons = inst.cons[:]
    rng.shuffle(cons)
    style = rng.choice(("agent", "user"))
    goal = f"the {dom.noun} {join([clause(c, rng) for c in cons])}"
    traps = " ".join(_traps(cons, rng, _NAV_TRAP))
    traps = (" " + traps) if traps else ""
    if style == "agent":
        return style, (f"Goal: open {goal}.{traps}\nCurrent page: {here}\n"
                       f"Page title: {rng.choice(dom.intro)}")
    return style, f"I'm on {here} right now. Can you take me to {goal}?{traps}"


def state_route(inst: Inst, rng: random.Random) -> tuple[str, str]:
    dom = inst.dom
    cons = inst.cons[:]
    rng.shuffle(cons)
    style = rng.choice(("ticket", "chat"))
    sents = [clause(c, rng) for c in cons]
    sents = [s[0].upper() + s[1:] if s else s for s in sents]
    parts = sents + _traps(cons, rng, _ROUTE_TRAP)
    rng.shuffle(parts)
    if style == "ticket":
        return style, (f"{rng.choice(dom.intro)} #{rng.randrange(10000, 99999)}\n"
                       f"{rng.choice(_GREET)} " + " ".join(parts))
    return style, "\n".join(f"Customer: {p}" for p in parts)


QUESTIONS = {
    "match": (("Which catalog entry meets every requirement?",
               "Which item satisfies all of the stated requirements?",
               "Which entry should be supplied?"),
              "Exactly one entry meets every requirement; each other entry misses at least "
              "one."),
    "link": (("Which knowledge-base entry is the passage about?",
              "Which entity does the mention refer to?",
              "Which candidate is the one meant here?"),
             "The candidates share the name. Exactly one matches every detail the passage "
             "gives about it."),
    "nav": (("Which link should the agent follow?",
             "Which link leads to the page the goal asks for?",
             "Which link should be clicked next?"),
            "Exactly one link matches every part of the goal."),
    "route": (("Which queue should this ticket be routed to?",
               "Which queue handles this request?",
               "Where should this ticket go?"),
              "Each queue covers one combination. Exactly one matches every fact in the "
              "ticket."),
}

# Option keys carry no information and cost few tokens: Qwen3 spends a token per digit,
# so "MB-7376" was 7 tokens a key -- 1,000 of a 150-option budget. Two syllables are 1-3.
_SYL = tuple(c + v for c in "bdfgklmnprstvz" for v in "aeiou")


def _url(dom: Domain, cell: dict) -> str:
    return dom.url.format(**{a.name: show(a, cell[a.name]) for a in dom.attrs})


def _keys(inst: Inst, rng: random.Random, name: str) -> list[str] | None:
    dom, n = inst.dom, inst.n
    if dom.key == "url":
        keys = [_url(dom, c) for c in inst.cells]
    else:
        keys, seen = [], set()
        while len(keys) < n:
            code = rng.choice(_SYL) + rng.choice(_SYL)
            if dom.key == "sku":
                k = code.upper()
            elif dom.key == "entity":
                k = f"{name} ({code})"
            else:
                k = code + "-desk"
            if k not in seen:
                seen.add(k)
                keys.append(k)
    if len({k.lower() for k in keys}) != len(keys):
        return None
    return keys


def _desc(dom: Domain, cell: dict) -> str:
    out = dom.desc
    for a in dom.attrs:
        out = out.replace("{" + a.name + "}", show(a, cell[a.name]))
    return out


def _here(inst: Inst, rng: random.Random, keys: list[str]) -> str | None:
    """nav: the page the agent is on. Off the option list and failing the goal, with each
    constrained value drawn uniformly from its axis -- so it names the gold's values no
    more often than any other entry's."""
    dom = inst.dom
    taken = set(keys)
    for _ in range(30):
        cell = {a.name: rng.choice(inst.axes[a.name]) if a.name in inst.axes
                else rng.choice(pool(a)) for a in dom.attrs}
        if all(c.ok(cell[c.attr.name]) for c in inst.cons):
            continue
        url = _url(dom, cell)
        if url not in taken:
            return url
    return None


def render(inst: Inst, rng: random.Random, qid: str, task: str,
           large: bool = False) -> Example | None:
    dom = inst.dom
    name = rng.choice(dom.names) if dom.names else ""
    keys = _keys(inst, rng, name)
    if keys is None:
        return None
    if dom.family == "match":
        style, state = state_match(inst, rng)
    elif dom.family == "link":
        style, state = state_link(inst, rng, name)
    elif dom.family == "nav":
        here = _here(inst, rng, keys)
        if here is None:
            return None
        style, state = state_nav(inst, rng, here)
    else:
        style, state = state_route(inst, rng)
    qs, instructions = QUESTIONS[dom.family]
    # a link whose URL already says everything carries no description (news archive)
    descs = [_desc(dom, c) or None for c in inst.cells] if dom.desc else None
    n = inst.n
    meta = {
        "gen": {"family": dom.family, "domain": dom.name, "n": n, "k": len(inst.cons),
                "hn_share": round(len(inst.hn) / (n - 1), 3) if n > 1 else 0.0,
                # every entry failing exactly one constraint, other blocks' included
                "near_share": round(sum(len(inst.fails(i)) == 1 for i in range(n))
                                    / (n - 1), 3) if n > 1 else 0.0,
                "blocks": inst.blocks, "shape": inst.shape, "style": style,
                "attrs": [c.attr.name for c in inst.cons],
                "ops": [c.op for c in inst.cons],
                "cases": [c.case for c in inst.cons]},
        "hard_neg": inst.hn,
        "hn_kind": inst.hn_kind,
    }
    if large:
        meta["large_n"] = True
    q = Question(id=qid, question=rng.choice(qs), options=keys, target=inst.gold,
                 meta=meta, instructions=instructions, descriptions=descs)
    return Example(state=state, questions=[q], task=task)


_DIGIT = re.compile(r"\d")


def est_tokens(text: str, cpt: float) -> float:
    d = len(_DIGIT.findall(text))
    return d + (len(text) - d) / cpt


def packed_tokens(ex: Example, cpt: float) -> tuple[float, float]:
    """(state, total) estimated tokens of the text the packer will tokenise."""
    q = ex.questions[0]
    head = f"\n\nQuestion: {q.question}\n{q.instructions}\nOptions:"
    descs = q.descriptions or [None] * len(q.options)
    opts = sum(est_tokens(f"\n- {o}: {d}" if d else f"\n- {o}", cpt)
               for o, d in zip(q.options, descs))
    s = est_tokens("State:\n" + ex.state, cpt)
    return s, s + est_tokens(head, cpt) + opts


def fits(ex: Example, cpt: float) -> bool:
    s, total = packed_tokens(ex, cpt)
    return s <= MAX_STATE_TOKENS - 64 and total <= FIT_TOKENS


def make(dom: Domain, rng: random.Random, qid: str, task: str,
         large: bool = False) -> tuple[Example, Inst] | None:
    """One example. The option count is drawn heavy-tailed, then narrowed until the
    rendered example fits the training budget (large-N tasks are exempt)."""
    n = draw_n(rng, LARGE_MIN, LARGE_MAX) if large else draw_n(rng)
    for _ in range(8):
        inst = build(dom, rng, n, large)
        if inst is None:
            continue
        ex = render(inst, rng, qid, task, large)
        if ex is None:
            continue
        if large:
            if inst.n >= LARGE_MIN:
                return ex, inst
            continue
        if fits(ex, dom.cpt):
            return ex, inst
        s, total = packed_tokens(ex, dom.cpt)
        per = (total - s) / max(1, inst.n)
        room = FIT_TOKENS - s - 60
        n = max(N_MIN, min(inst.n - 1, int(room / per)))
    return None


# ---- tasks -------------------------------------------------------------------------------

def instances(family: str, domains: tuple, seed: int, task: str, n: int,
              large: bool = False) -> Iterator[tuple[Example, Inst]]:
    """The loader's stream with the structure behind each example, for the audit."""
    made = 0
    for i in range(n * 4):
        if made >= n:
            return
        dom = domains[i % len(domains)]
        rng = random.Random(f"catalog:{seed}:{task}:{i}")
        got = make(dom, rng, f"{family}_{i}", task, large)
        if got is not None:
            made += 1
            yield got


def _loader(family: str, domains: tuple, seed: int, task: str, cap: int,
            large: bool = False):
    def load(n: int) -> Iterator[Example]:
        for ex, _ in instances(family, domains, seed, task, min(n, cap), large):
            yield ex
    # what `instances` needs to replay this stream with its structure (the audit, tests)
    load.spec = (family, domains, seed, task, large)
    return load


def tasks() -> list[RealTask]:
    """Per family: each trained domain to `train` with a fresh-seed `_eval` twin in
    `devreal`; one held-out domain to `devreal`, another to `testreal`, and a large-N
    task on the test domain. `family` names the held-out unit, the domain."""
    out: list[RealTask] = []

    def add(name, split, fam, doms, seed, cap, family, notes, large=False):
        out.append(RealTask(
            row=ROW, name=name, licence=LICENCE, url=URL,
            load=_loader(fam, doms, seed, name, cap, large), real=False,
            per_example_options=True, force_split=split, family=family, notes=notes))

    for fam, (trained, dev, test) in DOMAINS.items():
        for d in trained:
            base = f"catalog_{fam}_{d.name}"
            add(base, "train", fam, (d,), 0, TRAIN_CAP, base,
                f"{fam} over {d.name}; code-labelled, heavy-tailed option count")
            add(base + "_eval", "devreal", fam, (d,), 5051, TWIN_CAP, base,
                f"{fam} over {d.name}, fresh seed: in-structure twin of a trained domain")
        for d in dev:
            name = f"catalog_{fam}_{d.name}"
            add(name, "devreal", fam, (d,), 6061, HELDOUT_CAP, name,
                f"{fam} over {d.name}; domain held out of training (dev)")
        for d in test:
            name = f"catalog_{fam}_{d.name}"
            add(name, "testreal", fam, (d,), 7071, HELDOUT_CAP, name,
                f"{fam} over {d.name}; domain NEVER trained on (test)")
            add(f"catalog_{fam}_largeN", "testreal", fam, (d,), 8081, LARGE_CAP, name,
                f"{fam} over {d.name} with N 200-2,000; does not fit the packer, for "
                "sharded scoring only (meta.large_n)", large=True)
    return out
