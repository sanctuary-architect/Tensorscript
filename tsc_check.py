#!/usr/bin/env python3
"""
Reference validator for TensorScript v1.0 (.tensor) files.

Usage:
    python3 tsc_check.py path/to/file.tensor

Runs the file through the lexer, the recursive-descent parser (grammar in
tensorscript_v1_spec.md §5), and the cross-block semantic checks (§6).
Exits 0 and prints "OK" if the file is valid; otherwise prints the first
error found and exits 1.

This is a reference validator, not a production compiler — it checks that
a .tensor file is well-formed and internally consistent under the current
spec. It does not execute training runs.
"""

import sys
import json
from lexer import TensorScriptSyntaxError
from parser import parse
from semantic_checks import check


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 tsc_check.py path/to/file.tensor")
        sys.exit(2)

    path = sys.argv[1]
    with open(path) as f:
        source = f.read()

    try:
        ast = parse(source)
    except TensorScriptSyntaxError as e:
        print(f"SYNTAX ERROR in {path}:")
        print(f"  {e}")
        sys.exit(1)

    errors = check(ast)
    if errors:
        print(f"SEMANTIC ERROR(S) in {path}:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    block_summary = ", ".join(
        f"{b['type']}:{b.get('name', b.get('target', ''))}" for b in ast["blocks"]
    )
    print(f"OK — {path} is valid TensorScript v{ast['version']}")
    print(f"  version: {ast['version']}")
    print(f"  imports: {len(ast['imports'])}")
    print(f"  blocks:  {block_summary}")


if __name__ == "__main__":
    main()
