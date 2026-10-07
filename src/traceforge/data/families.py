"""Realistic input families for the synthetic task generator.

Every sampler takes a ``random.Random`` and returns one input row (tuple of strings).  The
word lists are small, hand-written and synthetic (no external data).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

FIRST = (
    "Ada Grace Alan Linus Margaret Dennis Barbara Edsger Katherine Donald Frances Ken Radia Tim Hedy "
    "Vint Shafi Leslie Sophie Niklaus Anita Brian Joan Bjarne Guido Yukihiro Lynn Whitfield Jean "
    "Marissa Satya Sundar Fei Andrew Demis Geoffrey Yann Judea Cynthia Timnit Daphne Pieter"
).split()
LAST = (
    "Lovelace Hopper Turing Torvalds Hamilton Ritchie Liskov Dijkstra Johnson Knuth Allen Thompson "
    "Perlman Berners Lamarr Cerf Goldwasser Lamport Wilson Wirth Borg Kernighan Clarke Stroustrup "
    "Rossum Matsumoto Conway Diffie Sammet Mayer Nadella Pichai Li Ng Hassabis Hinton LeCun Pearl "
    "Dwork Gebru Koller Norvig"
).split()
WORDS = (
    "report summary budget invoice draft final notes plan review backup archive photo travel "
    "project alpha beta gamma delta sales q1 q2 data model train test config readme"
).split()
COMPANIES = (
    "Analytical Engines|Bletchley Park|Bell Labs|Xerox Parc|Sun Micro|Open Source Co|Cray Research|"
    "Acme Corp|Initech|Globex|Umbrella Labs|Hooli"
).split("|")
DOMAINS = ("example.com", "mail.org", "uni.edu", "corp.net", "dev.io", "post.co.uk", "navy.mil", "lab.ai")
EXTS = ("txt", "csv", "py", "md", "json", "pdf", "png")
DIRS = ("home", "usr", "var", "data", "src", "docs", "tmp", "opt", "etc", "work")
PREFIXES = ("INV", "ORD", "SKU", "TKT", "ID", "REF", "PO")
CITIES = ("austin", "boston", "denver", "seattle", "portland", "san jose", "new york", "las vegas", "miami")
STATES = ("tx", "ma", "co", "wa", "or", "ca", "ny", "nv", "fl")


def _noisy(s: str, rng: random.Random, pad: float = 0.2, case: float = 0.12) -> str:
    r = rng.random()
    if r < case / 2:
        s = s.lower()
    elif r < case:
        s = s.upper()
    if rng.random() < pad:
        s = " " * rng.randint(1, 2) + s
    if rng.random() < pad:
        s = s + " " * rng.randint(1, 2)
    return s


def _name_pair(rng: random.Random) -> tuple[str, ...]:
    return _noisy(rng.choice(FIRST), rng, 0.15), _noisy(rng.choice(LAST), rng, 0.15)


def _full_name(rng: random.Random) -> tuple[str, ...]:
    parts = [rng.choice(FIRST), rng.choice(LAST)]
    if rng.random() < 0.2:
        parts.insert(1, rng.choice("ABCDEFGHJKLMRST") + ".")
    return (_noisy(" ".join(parts), rng, 0.25),)


def _last_first(rng: random.Random) -> tuple[str, ...]:
    return (_noisy(f"{rng.choice(LAST)}, {rng.choice(FIRST)}", rng, 0.15),)


def _email(rng: random.Random) -> tuple[str, ...]:
    sep = rng.choice((".", ".", "_"))
    local = f"{rng.choice(FIRST)}{sep}{rng.choice(LAST)}".lower()
    if rng.random() < 0.15:
        local = rng.choice(FIRST).lower()
    return (_noisy(f"{local}@{rng.choice(DOMAINS)}", rng, 0.1, 0.1),)


def _identifier(rng: random.Random) -> tuple[str, ...]:
    sep = rng.choice(("_", "-"))
    k = rng.randint(2, 4)
    return (sep.join(rng.choice(WORDS) for _ in range(k)),)


def _path(rng: random.Random) -> tuple[str, ...]:
    d = "/".join(rng.choice(DIRS) for _ in range(rng.randint(1, 3)))
    f = rng.choice(WORDS)
    if rng.random() < 0.25:
        f += "_" + rng.choice(WORDS)
    return (f"/{d}/{f}.{rng.choice(EXTS)}",)


def _code(rng: random.Random) -> tuple[str, ...]:
    p = rng.choice(PREFIXES)
    if rng.random() < 0.5:
        return (f"{p}-{rng.randint(2019, 2025)}-{rng.randint(1, 9999):04d}",)
    return (f"{p}-{rng.randint(1, 99999)}",)


def _date(rng: random.Random) -> tuple[str, ...]:
    y, m, d = rng.randint(1999, 2025), rng.randint(1, 12), rng.randint(1, 28)
    return (rng.choice((f"{y}-{m:02d}-{d:02d}", f"{d:02d}/{m:02d}/{y}")),)


def _phrase(rng: random.Random) -> tuple[str, ...]:
    words = [rng.choice(WORDS) for _ in range(rng.randint(2, 4))]
    words = [w.capitalize() if rng.random() < 0.4 else w for w in words]
    return (_noisy(" ".join(words), rng, 0.25, 0.1),)


def _name_company(rng: random.Random) -> tuple[str, ...]:
    return _noisy(f"{rng.choice(FIRST)} {rng.choice(LAST)}", rng, 0.1), rng.choice(COMPANIES)


def _padded(rng: random.Random) -> tuple[str, ...]:
    base = rng.choice(WORDS) + (rng.choice((" ", "_", "-")) + rng.choice(WORDS) if rng.random() < 0.5 else "")
    if rng.random() < 0.3:
        base = base.title()
    return (" " * rng.randint(0, 3) + base + " " * rng.randint(0, 3),)


def _city_state(rng: random.Random) -> tuple[str, ...]:
    return _noisy(rng.choice(CITIES), rng, 0.15, 0.1), rng.choice(STATES)


def _user_domain(rng: random.Random) -> tuple[str, ...]:
    return f"{rng.choice(FIRST)}.{rng.choice(LAST)[:3]}".lower(), rng.choice(DOMAINS)


@dataclass(frozen=True)
class Family:
    name: str
    columns: tuple[str, ...]
    delims: tuple[str, ...]
    sample: Callable[[random.Random], tuple[str, ...]]


FAMILIES: tuple[Family, ...] = (
    Family("name_pair", ("first", "last"), (), _name_pair),
    Family("full_name", ("full_name",), (" ", "."), _full_name),
    Family("last_first", ("name",), (",", " "), _last_first),
    Family("email", ("email",), ("@", ".", "_"), _email),
    Family("identifier", ("ident",), ("_", "-"), _identifier),
    Family("path", ("path",), ("/", ".", "_"), _path),
    Family("code", ("code",), ("-",), _code),
    Family("date", ("date",), ("-", "/"), _date),
    Family("phrase", ("text",), (" ",), _phrase),
    Family("name_company", ("name", "company"), (" ",), _name_company),
    Family("padded", ("value",), (" ", "_", "-"), _padded),
    Family("city_state", ("city", "state"), (" ",), _city_state),
    Family("user_domain", ("user", "domain"), (".",), _user_domain),
)
FAMILY_BY_NAME = {f.name: f for f in FAMILIES}
