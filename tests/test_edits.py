from terminal_agent.edits import apply_edit, detect_eol

SRC = "def f(x):\n    return x + 1\n\n\ndef g(x):\n    return x + 1\n"


def test_unique_exact_match_replaces_and_reports_line():
    out = apply_edit(SRC, "def g(x):\n    return x + 1\n", "def g(x):\n    return x + 2\n")
    assert out.ok and out.strategy == "exact" and out.line == 5
    assert out.content is not None and out.content.endswith("return x + 2\n")
    assert out.content.count("return x + 1") == 1


def test_ambiguous_match_changes_nothing_and_counts():
    out = apply_edit(SRC, "    return x + 1\n", "    return x\n")
    assert out.status == "ambiguous" and out.occurrences == 2 and out.content is None


def test_expected_replacements_allows_replace_all():
    out = apply_edit(SRC, "x + 1", "x + 3", expected=2)
    assert out.ok and out.content is not None and out.content.count("x + 3") == 2


def test_not_found_and_noop_and_empty():
    assert apply_edit(SRC, "nope", "x").status == "not_found"
    assert apply_edit(SRC, "def f", "def f").status == "noop"
    assert apply_edit(SRC, "", "x").status == "empty_old"


def test_crlf_file_matches_lf_snippet_and_stays_crlf():
    crlf = SRC.replace("\n", "\r\n")
    out = apply_edit(crlf, "def f(x):\n    return x + 1\n", "def f(x):\n    return x - 1\n")
    assert out.ok and out.strategy == "crlf"
    assert out.content is not None and "\n" not in out.content.replace("\r\n", "")
    assert "return x - 1\r\n" in out.content


def test_trailing_whitespace_slip_needs_fuzzy():
    src = "a = 1   \nb = 2\n"
    assert apply_edit(src, "a = 1\nb = 2\n", "a = 9\nb = 2\n").status == "not_found"
    out = apply_edit(src, "a = 1\nb = 2\n", "a = 9\nb = 2\n", fuzzy=True)
    assert out.ok and out.strategy == "rstrip" and out.content == "a = 9\nb = 2\n"


def test_indent_fuzzy_reindents_replacement():
    src = "class A:\n    def m(self):\n        return 1\n"
    out = apply_edit(src, "def m(self):\n    return 1\n", "def m(self):\n    return 2\n",
                     fuzzy=True)
    assert out.ok and out.strategy == "indent"
    assert out.content == "class A:\n    def m(self):\n        return 2\n"


def test_fuzzy_still_refuses_ambiguity():
    src = "if a:\n    x = 1\nif b:\n        x = 1\n"
    assert apply_edit(src, "x = 1 \n", "x = 2\n").status == "not_found"
    out = apply_edit(src, "x = 1 \n", "x = 2\n", fuzzy=True)
    assert out.status == "ambiguous" and out.strategy == "indent"


def test_detect_eol():
    assert detect_eol("a\r\nb\r\n") == "\r\n"
    assert detect_eol("a\nb") == "\n"
    assert detect_eol("") == "\n"


def test_indent_fuzzy_shifts_lines_indented_less_than_the_first():
    # a dedented snippet whose first line is nested deeper than a later line: the first
    # fuzzy implementation left the later line at column 0 (7 of 10 real hunks)
    src = "def f(x):\n    if x:\n        return 1\n    raise ValueError(x)\n"
    old = "    return 1\nraise ValueError(x)\n"
    new = "    return 2\nraise TypeError(x)\n"
    out = apply_edit(src, old, new, fuzzy=True)
    assert out.ok and out.strategy == "indent"
    assert out.content == "def f(x):\n    if x:\n        return 2\n    raise TypeError(x)\n"


def test_fuzzy_indent_refuses_a_structurally_different_snippet():
    # a line the model misremembers as inside the `if` must NOT match a file where it is
    # outside (reviewer issue 8: the old str.strip norm ignored relative indentation)
    src = "def f(a):\n    if a:\n        return 1\n    return 2\n"
    old = "if a:\n    return 1\n    return 2\n"      # return 2 wrongly nested in the if
    new = "if a:\n    return 10\n    return 2\n"
    assert apply_edit(src, old, new, fuzzy=True).status == "not_found"


def test_fuzzy_indent_does_not_merge_tabs_and_spaces():
    src = "class A:\n\tdef g(self):\n\t\tif x:\n\t\t\treturn 1\n\t\treturn 2\n"
    old = "    def g(self):\n        if x:\n            return 1\n        return 2\n"
    new = "    def g(self):\n        if x:\n            return 3\n        return 2\n"
    assert apply_edit(src, old, new, fuzzy=True).status == "not_found"


def test_fuzzy_indent_still_applies_a_uniformly_dedented_block():
    src = "class A:\n    def m(self):\n        return 1\n"
    out = apply_edit(src, "def m(self):\n    return 1\n", "def m(self):\n    return 2\n",
                     fuzzy=True)
    assert out.ok and out.strategy == "indent"
    assert out.content == "class A:\n    def m(self):\n        return 2\n"
