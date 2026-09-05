"""
Tokenizer for TensorScript v1.0.

Produces a flat token stream from raw source text. Comments (# ...) are
stripped. Strings, numbers, percents, identifiers, and punctuation are
each tokenized distinctly so the parser never has to re-inspect raw text.
"""

import re
from dataclasses import dataclass


class TensorScriptSyntaxError(Exception):
    def __init__(self, message, line, col):
        self.line = line
        self.col = col
        super().__init__(f"Syntax error at line {line}, col {col}: {message}")


@dataclass
class Token:
    kind: str   # STRING, NUMBER, PERCENT, IDENT, KEYWORD, PUNCT, EOF
    value: str
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.kind!r}, {self.value!r}, {self.line}:{self.col})"


KEYWORDS = {
    "tensorscript", "import", "as", "dataset", "streams", "model", "monitor",
    "optimize", "pipeline", "stage", "evaluate", "using", "telemetry",
    "schedule", "epochs", "seed", "hardware", "guardrails", "if", "for",
    "flat_lines", "env", "scratch", "checkpoint", "best", "final", "step",
    "minimize", "maximize", "every_checkpoint", "final_only", "on",
    "benchmarks", "after", "report_to", "run", "depends_on",
    "inherit_weights", "true", "false", "none",
}

PUNCT_MULTI = [":", "{", "}", "[", "]", "(", ")", ",", ".", "%", ">=", "<=",
               "==", ">", "<"]
# order matters: longer operators first
PUNCT_MULTI.sort(key=len, reverse=True)

TOKEN_SPEC = [
    ("SKIP",    r"[ \t\r]+"),
    ("NEWLINE", r"\n"),
    ("COMMENT", r"#[^\n]*"),
    ("STRING",  r'"(?:[^"\\]|\\.)*"'),
    ("PERCENT", r"\d+(?:\.\d+)?%"),
    ("QUANTBIT", r"\d+bit\b"),  # 4bit / 8bit / 16bit: digit-led, so neither NUMBER nor IDENT can claim it on their own
    ("NUMBER",  r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"),
    ("IDENT",   r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP2",     r">=|<=|=="),
    ("OP1",     r"[:{}\[\]().,%><=\-]"),
    ("EMOJI_OR_OTHER", r"[^\sA-Za-z0-9_\"#]"),  # tolerate stray unicode e.g. in print() strings (already captured by STRING though)
]

MASTER_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))


def tokenize(source: str):
    tokens = []
    line = 1
    line_start = 0
    pos = 0
    length = len(source)

    while pos < length:
        m = MASTER_RE.match(source, pos)
        if not m:
            col = pos - line_start + 1
            raise TensorScriptSyntaxError(f"unrecognized character {source[pos]!r}", line, col)
        kind = m.lastgroup
        text = m.group()
        col = m.start() - line_start + 1

        if kind == "NEWLINE":
            line += 1
            line_start = m.end()
        elif kind in ("SKIP", "COMMENT"):
            pass
        elif kind == "STRING":
            tokens.append(Token("STRING", text[1:-1], line, col))
        elif kind == "PERCENT":
            tokens.append(Token("PERCENT", text[:-1], line, col))
        elif kind == "NUMBER":
            tokens.append(Token("NUMBER", text, line, col))
        elif kind == "QUANTBIT":
            tokens.append(Token("IDENT", text, line, col))
        elif kind == "IDENT":
            if text in KEYWORDS:
                tokens.append(Token("KEYWORD", text, line, col))
            else:
                tokens.append(Token("IDENT", text, line, col))
        elif kind in ("OP1", "OP2"):
            tokens.append(Token("PUNCT", text, line, col))
        elif kind == "EMOJI_OR_OTHER":
            # Allow stray unicode (e.g. emoji) only inside strings; outside strings it's an error.
            raise TensorScriptSyntaxError(f"unexpected character {text!r}", line, col)
        pos = m.end()

    tokens.append(Token("EOF", "", line, pos - line_start + 1))
    return tokens
