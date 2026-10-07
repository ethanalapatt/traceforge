"""Independently authored data-cleaning tasks.

Supplied rows (input -> output) are written by hand.  Hidden rows are hand-written *inputs*
chosen to separate plausible shortcuts (e.g. a task whose supplied rows are consistent with both
``Slice`` and ``Token``); their expected outputs come from the authored reference program, which
``tests/test_data.py`` verifies against every hand-written supplied output.  Reference programs are
evaluator-only: the solver never sees them.

The first six tasks are tagged ``regression`` and form the smoke-profile evaluation set.
"""

from __future__ import annotations

from .. import dsl
from ..dsl import Concat, Input, Literal, Lower, Replace, Slice, Token, Trim, Upper
from ..tasks import EvalTask, Example, TaskSpec

I0, I1 = Input(0), Input(1)
L = Literal


def tok(a, d, i):
    return Token(a, d, i)


def sl(a, s, e):
    return Slice(a, s, e)


def rep(a, o, n):
    return Replace(a, o, n)


def _one(*pairs):
    return [((a,), b) for a, b in pairs]


# (id, columns, supplied [(inputs, output)], hidden input rows, reference AST)
_RAW: list[tuple] = [
    # ------------------------------------------------------------- regression (smoke) set
    ("trim_only", ("name",), _one(("  Ada ", "Ada"), ("Bob  ", "Bob"), (" Cy", "Cy"), ("dee", "dee")),
     [("  x  ",), ("\tTab\t",), ("mid dle ",), ("   ",), ("",), (" ünï ",), ("a  b",), (" Z",)], Trim(I0)),
    ("upper_code", ("code",), _one(("ab12", "AB12"), ("x9", "X9"), ("Qz7k", "QZ7K"), ("mn", "MN")),
     [("aBc",), ("123",), ("Straße",), ("é",), ("",), ("a-b_c",), ("MiXeD",)], Upper(I0)),
    ("email_domain", ("email",), _one(("ada@analytical.org", "analytical.org"), ("grace@navy.mil", "navy.mil"),
                                      ("alan@bletchley.co.uk", "bletchley.co.uk"), ("linus@kernel.org", "kernel.org")),
     [("x@y.z",), ("grace.hopper@navy.mil",), ("a@b.co.uk",), ("first_last@sub.example.com",), ("u@h",)], tok(I0, "@", 1)),
    ("file_extension", ("filename",), _one(("report.pdf", "pdf"), ("notes.txt", "txt"), ("photo.final.png", "png"), ("a.b", "b")),
     [("archive.tar.gz",), ("x.y.z.w",), ("readme.md",), ("data.v2.csv",), ("my.file.name.json",)], tok(I0, ".", -1)),
    ("normalize_name", ("full_name",), _one(("  Ada Lovelace  ", "ada.lovelace"), ("Grace Hopper", "grace.hopper"),
                                            ("  Alan Turing", "alan.turing"), ("MARGARET HAMILTON ", "margaret.hamilton")),
     [("  Katherine Johnson ",), ("DOROTHY VAUGHAN",), ("mary jackson",), ("Annie  Easley",), (" Radia Perlman",),
      ("Ken Thompson ",), ("Barbara Liskov",)], Lower(rep(Trim(I0), " ", "."))),
    ("last_first", ("first", "last"), [(("Ada", "Lovelace"), "Lovelace, Ada"), (("Grace", "Hopper"), "Hopper, Grace"),
                                        (("Alan", "Turing"), "Turing, Alan"), (("Edsger", "Dijkstra"), "Dijkstra, Edsger")],
     [("Mary Jane", "Watson-Parker"), ("ada", "lovelace"), ("X", "Y"), ("Jean", "Bartik"), ("O'Neil", "Shaq"), ("Ünal", "Çelik")],
     Concat(I1, Concat(L(", "), I0))),
    # ----------------------------------------------------------------- remaining authored
    ("initial", ("name",), _one(("ada", "A"), ("grace", "G"), ("Alan", "A"), ("linus", "L")),
     [("émile",), ("x",), ("Zed",), ("mARY",), ("b",), ("9lives",)], Upper(sl(I0, 0, 1))),
    ("email_user", ("email",), _one(("ada@analytical.org", "ada"), ("grace@navy.mil", "grace"),
                                    ("alan@bletchley.co.uk", "alan"), ("linus@kernel.org", "linus")),
     [("first.last@x.org",), ("a_b@c.d",), ("solo@z",), ("dotted.name.here@host.net",), ("x@y",)], tok(I0, "@", 0)),
    ("drop_first_char", ("tag",), _one(("#42", "42"), ("#7", "7"), ("#1001", "1001"), ("#9", "9")),
     [("#abc",), ("##2",), ("#",), ("# 5",), ("#12345",), ("#a-b",)], sl(I0, 1, None)),
    ("year_from_date", ("date",), _one(("2024-03-17", "2024"), ("1999-12-31", "1999"), ("2010-07-04", "2010"), ("2001-01-09", "2001")),
     [("2000-1-1",), ("2024-03-17T10:00",), ("1987-05-22",), ("2031-11-30",), ("0001-01-01",)], tok(I0, "-", 0)),
    ("day_from_date", ("date",), _one(("2024-03-17", "17"), ("1999-12-31", "31"), ("2010-07-04", "04"), ("2001-01-09", "09")),
     [("2024-03-17T10:00",), ("2000-1-1",), ("1987-05-22",), ("2031-11-30",), ("2024-02-29",)], tok(I0, "-", -1)),
    ("snake_to_title_words", ("field",), _one(("user_id", "USER ID"), ("first_name", "FIRST NAME"),
                                              ("order_item_count", "ORDER ITEM COUNT"), ("zip_code", "ZIP CODE")),
     [("a_b_c",), ("__x__",), ("single",), ("with_digits_42",), ("trailing_",)], Upper(rep(I0, "_", " "))),
    ("domain_name_only", ("email",), _one(("ada@analytical.org", "analytical"), ("grace@navy.mil", "navy"),
                                          ("alan@bletchley.co.uk", "bletchley"), ("linus@kernel.org", "kernel")),
     [("x@y",), ("a@b.c.d.e",), ("user@sub.domain.com",), ("q@localhost",), ("z@www.site.io",)], tok(tok(I0, "@", 1), ".", 0)),
    ("user_at_site", ("email",), _one(("ada@analytical.org", "ada at analytical.org"), ("grace@navy.mil", "grace at navy.mil"),
                                      ("alan@bletchley.co.uk", "alan at bletchley.co.uk"), ("linus@kernel.org", "linus at kernel.org")),
     [("a@b",), ("x.y@z.w",), ("me@home",), ("first.last@corp.example.com",), ("solo@here.io",)],
     Concat(tok(I0, "@", 0), Concat(L(" at "), tok(I0, "@", 1)))),
    ("file_stem", ("path",), _one(("/home/ada/report.pdf", "report"), ("/tmp/notes.txt", "notes"),
                                  ("/var/log/syslog.1", "syslog"), ("docs/readme.md", "readme")),
     [("/a/b/c.tar.gz",), ("/noext",), ("photo.png",), ("/x/y/z/deep.file.name",), ("/etc/hosts",)], tok(tok(I0, "/", -1), ".", 0)),
    ("last_name_upper", ("name",), _one(("Ada Lovelace", "LOVELACE"), ("Grace Brewster Hopper", "HOPPER"),
                                        ("Alan Turing", "TURING"), ("Linus Torvalds", "TORVALDS")),
     [("Mary Ann Smith",), ("Cher",), ("van der Berg",), ("Jean-Luc Picard",), ("O Brien",)], Upper(tok(I0, " ", -1))),
    ("first_name_title", ("name",), _one(("ADA LOVELACE", "Ada"), ("grace hopper", "Grace"),
                                         ("aLAN turing", "Alan"), ("Linus Torvalds", "Linus")),
     [("mARGARET hamilton",), ("x y",), ("JEAN  BARTIK",), ("éMILE zola",), ("single",), ("RADIA perlman",)],
     Concat(Upper(sl(tok(I0, " ", 0), 0, 1)), sl(Lower(tok(I0, " ", 0)), 1, None))),
    ("phone_digits", ("phone",), _one(("555-123-4567", "5551234567"), ("800-555-0199", "8005550199"),
                                      ("212-867-5309", "2128675309"), ("415-555-0101", "4155550101")),
     [("555-1234",), ("1-800-555-0199",), ("000-000-0000",), ("12-34",), ("555-123-4567-89",)], rep(I0, "-", "")),
    ("phone_area_code", ("phone",), _one(("(555) 123-4567", "555"), ("(800) 555-0199", "800"),
                                         ("(212) 867-5309", "212"), ("(415) 555-0101", "415")),
     [("(1) 2",), ("(0000) 111",), ("(917) 555-2368",), ("(7) 555-0000",), ("(65) 1",)], tok(tok(I0, ")", 0), "(", 1)),
    ("initials_two_cols", ("first", "last"), [(("Ada", "Lovelace"), "AL"), (("grace", "hopper"), "GH"),
                                               (("Alan", "turing"), "AT"), (("linus", "Torvalds"), "LT")],
     [("mary-jane", "watson"), ("x", "y"), ("Émile", "Zola"), ("Jean", "Bartik"), ("o'neil", "Shaq")],
     Concat(Upper(sl(I0, 0, 1)), Upper(sl(I1, 0, 1)))),
    ("city_state", ("city", "state"), [(("Austin", "tx"), "Austin, TX"), (("Boston", "ma"), "Boston, MA"),
                                        (("Denver", "co"), "Denver, CO"), (("Seattle", "wa"), "Seattle, WA")],
     [("new york", "ny"), ("san jose", "ca"), ("MIAMI", "fl"), ("portland", "or"), ("Las Vegas", "nv")],
     Concat(I0, Concat(L(", "), Upper(I1)))),
    ("strip_id_prefix", ("id",), _one(("N-1234", "1234"), ("N-77", "77"), ("N-30518", "30518"), ("N-9", "9")),
     [("N-12-34",), ("N-007-A",), ("N-5",), ("N-100000",), ("N-1-2-3",)], sl(I0, 2, None)),
    ("domain_upper", ("email",), _one(("ada@analytical.org", "ANALYTICAL.ORG"), ("grace@navy.mil", "NAVY.MIL"),
                                      ("alan@bletchley.co.uk", "BLETCHLEY.CO.UK"), ("linus@kernel.org", "KERNEL.ORG")),
     [("x@y.z",), ("a.b@c.d.e",), ("u@v",), ("someone@Mixed.Case.Com",), ("q@w.io",)], Upper(tok(I0, "@", 1))),
    ("username_clean", ("name",), _one((" Ada Lovelace ", "ada_lovelace"), ("GRACE HOPPER", "grace_hopper"),
                                       ("alan turing", "alan_turing"), ("  Linus Torvalds", "linus_torvalds")),
     [("Alan  Turing",), ("Single",), (" X Y ",), ("Mary Ann Smith",), ("  AB CD  ",)], Lower(rep(Trim(I0), " ", "_"))),
    ("email_from_name", ("full_name",), _one(("Ada Lovelace", "ada@example.com"), ("Grace Hopper", "grace@example.com"),
                                             ("Alan Turing", "alan@example.com"), ("Linus Torvalds", "linus@example.com")),
     [("Mary Ann Smith",), ("Cher",), ("JEAN BARTIK",), ("radia perlman",), ("Ken Thompson",)],
     Concat(Lower(tok(I0, " ", 0)), L("@example.com"))),
    ("dotted_email", ("full_name",), _one(("Ada Lovelace", "ada.lovelace"), ("Grace Hopper", "grace.hopper"),
                                          ("Alan Turing", "alan.turing"), ("Linus Torvalds", "linus.torvalds")),
     [("Mary Ann Smith",), ("Grace Brewster Hopper",), ("Cher Sarkisian",), ("Jean Luc Picard",), ("Ken Thompson",)],
     Concat(Lower(tok(I0, " ", 0)), Concat(L("."), Lower(tok(I0, " ", -1))))),
    ("base_name_upper", ("path",), _one(("/home/ada/report.pdf", "REPORT"), ("/tmp/notes.txt", "NOTES"),
                                        ("/var/log/syslog.1", "SYSLOG"), ("docs/readme.md", "README")),
     [("/a/b/c.tar.gz",), ("photo.png",), ("/x/y/deep.file.name",), ("/etc/hosts",), ("/tmp/.hidden",)],
     Upper(tok(tok(I0, "/", -1), ".", 0))),
    ("extension_upper", ("filename",), _one(("report.pdf", "PDF"), ("notes.txt", "TXT"), ("photo.final.png", "PNG"), ("a.b", "B")),
     [("archive.tar.gz",), ("x.y.z.w",), ("readme.md",), ("data.v2.csv",), ("noext",)], Upper(tok(I0, ".", -1))),
    ("flip_comma_name", ("name",), _one(("Hopper, Grace", "Grace Hopper"), ("Turing, Alan", "Alan Turing"),
                                        ("Lovelace, Ada", "Ada Lovelace"), ("Torvalds, Linus", "Linus Torvalds")),
     [("Watson-Parker, Mary Jane",), ("Doe,John",), ("Smith, J.",), ("Zola, Émile",), ("Single,",)],
     Concat(Trim(tok(I0, ",", 1)), Concat(L(" "), tok(I0, ",", 0)))),
    ("csv_cell_clean", ("cell",), _one(("  USER_ID ", "user id"), ("First_Name", "first name"),
                                       (" zip_CODE  ", "zip code"), ("ORDER_item_COUNT", "order item count")),
     [("__X__",), (" a_b ",), ("NoUnderscore",), ("A_B_C_D",), ("  ",)], Lower(Trim(rep(I0, "_", " ")))),
    ("ticket_id", ("number",), _one(("10432", "TKT-432"), ("88", "TKT-88"), ("7001", "TKT-001"), ("123", "TKT-123")),
     [("1",), ("999999",), ("00012",), ("5050",), ("42",)], Concat(L("TKT-"), sl(I0, -3, None))),
    ("swap_first_last", ("name",), _one(("Ada Lovelace", "Lovelace Ada"), ("Grace Hopper", "Hopper Grace"),
                                        ("Alan Turing", "Turing Alan"), ("Linus Torvalds", "Torvalds Linus")),
     [("Mary Ann Smith",), ("Cher",), ("Jean Luc",), ("A B C D",), ("Ken  Thompson",)],
     Concat(tok(I0, " ", -1), Concat(L(" "), tok(I0, " ", 0)))),
    ("repo_name", ("repo",), _one(("github.com/ada/engine", "engine"), ("gitlab.com/grace/compiler", "compiler"),
                                  ("example.org/alan/bombe", "bombe"), ("github.com/linus/linux", "linux")),
     [("github.com/a/b/c",), ("solo",), ("x/y",), ("host/path/to/thing",), ("trailing/",)], tok(I0, "/", -1)),
    ("shout_id", ("code",), _one(("abcdef", "ABC!"), ("xyz123", "XYZ!"), ("hello", "HEL!"), ("qwerty", "QWE!")),
     [("ab",), ("a",), ("12345",), ("mixedCase",), ("éclair",)], Concat(Upper(sl(I0, 0, 3)), L("!"))),
    ("quote_wrap", ("text",), _one(("  hello ", '"hello"'), ("world", '"world"'), (" a b", '"a b"'), ("x  ", '"x"')),
     [("  ",), ("it's",), ('say "hi"',), (" tab\t",), ("ünï",)], Concat(L('"'), Concat(Trim(I0), L('"')))),
    ("bracket_id", ("id",), _one(("abc", "[abc]"), ("42", "[42]"), ("x-y", "[x-y]"), ("Q", "[Q]")),
     [("",), ("a b",), ("[x]",), ("12345",), ("é",)], Concat(L("["), Concat(I0, L("]")))),
    ("dash_to_underscore_lower", ("slug",), _one(("Hello-World", "hello_world"), ("A-B-C", "a_b_c"),
                                                 ("Mixed-Case-Words", "mixed_case_words"), ("NODASH", "nodash")),
     [("--x--",), ("a-b",), ("Ünï-Cödé",), ("-",), ("ALL-CAPS-HERE-NOW",)], Lower(rep(I0, "-", "_"))),
    ("first_initial_last", ("first", "last"), [(("Ada", "Lovelace"), "A Lovelace"), (("Grace", "Hopper"), "G Hopper"),
                                                (("Alan", "Turing"), "A Turing"), (("Linus", "Torvalds"), "L Torvalds")],
     [("mary", "Watson Parker"), ("x", "y"), ("Émile", "Zola"), ("  Jean", "Bartik"), ("o'neil", "SHAQ")],
     Concat(sl(I0, 0, 1), Concat(L(" "), I1))),
    ("slugify", ("title",), _one((" Hello World_Foo ", "hello-world-foo"), ("My_Great Post", "my-great-post"),
                                 ("  A B_C", "a-b-c"), ("Plain Text_Here ", "plain-text-here")),
     [("One",), ("Two  Spaces_Here",), (" _lead",), ("a_b_c d",), ("  ",)],
     Lower(rep(rep(Trim(I0), " ", "-"), "_", "-"))),
    ("user_at_domain", ("user", "domain"), [(("ada.lov", "example.com"), "ada@example.com"),
                                             (("grace.hop", "navy.mil"), "grace@navy.mil"),
                                             (("alan.tur", "bletchley.org"), "alan@bletchley.org"),
                                             (("linus.tor", "kernel.org"), "linus@kernel.org")],
     [("x", "y.z"), ("a.b.c", "d.e"), ("solo", "host"), ("first.last", "mail.example.com"), (".lead", "x.io")],
     Concat(tok(I0, ".", 0), Concat(L("@"), I1))),
]

REGRESSION_IDS = tuple(r[0] for r in _RAW[:6])


def authored_tasks() -> list[EvalTask]:
    out: list[EvalTask] = []
    for tid, cols, supplied, hidden_in, ref in _RAW:
        spec = TaskSpec(
            f"authored-{tid}",
            tuple(cols),
            tuple(Example(tuple(i), o) for i, o in supplied),
        )
        hidden = []
        for row in hidden_in:
            v = dsl.evaluate(ref, row)
            assert isinstance(v, str), f"authored task {tid}: reference invalid on hidden row {row!r}"
            hidden.append(Example(tuple(row), v))
        out.append(
            EvalTask(
                spec, tuple(hidden), ref, "authored", dsl.skeleton(ref), "authored",
                {"ref_size": dsl.ast_size(ref), "regression": tid in REGRESSION_IDS, "authored_name": tid},
            )
        )
    return out


def regression_tasks() -> list[EvalTask]:
    return [t for t in authored_tasks() if t.meta["regression"]]
