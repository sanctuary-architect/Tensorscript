"""
Recursive-descent parser for TensorScript v1.0, implementing the grammar
in tensorscript_v1_spec.md §5.

Produces a plain-dict AST (kept deliberately simple/inspectable rather than
a class hierarchy, since the goal here is to validate the grammar itself,
not to ship a production compiler).
"""

from lexer import tokenize, Token, TensorScriptSyntaxError


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    # ---- token cursor helpers ----

    def peek(self, offset=0):
        return self.tokens[self.i + offset]

    def advance(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def check_kw(self, kw):
        tok = self.peek()
        return tok.kind == "KEYWORD" and tok.value == kw

    def check_punct(self, p):
        tok = self.peek()
        return tok.kind == "PUNCT" and tok.value == p

    def expect_kw(self, kw):
        if not self.check_kw(kw):
            tok = self.peek()
            raise TensorScriptSyntaxError(f"expected keyword {kw!r}, got {tok.value!r}", tok.line, tok.col)
        return self.advance()

    def expect_punct(self, p):
        if not self.check_punct(p):
            tok = self.peek()
            raise TensorScriptSyntaxError(f"expected {p!r}, got {tok.value!r}", tok.line, tok.col)
        return self.advance()

    def expect_ident_like(self):
        """Accepts IDENT or KEYWORD-as-identifier (TensorScript keywords can
        double as block field names in value position, e.g. metric names)."""
        tok = self.peek()
        if tok.kind in ("IDENT", "KEYWORD"):
            return self.advance()
        raise TensorScriptSyntaxError(f"expected identifier, got {tok.value!r}", tok.line, tok.col)

    # ---- top level ----

    def parse_file(self):
        version = self.parse_version_pin()
        imports = []
        while self.check_kw("import"):
            imports.append(self.parse_import_stmt())
        blocks = []
        while self.peek().kind != "EOF":
            blocks.append(self.parse_block())
        return {"version": version, "imports": imports, "blocks": blocks}

    def parse_version_pin(self):
        self.expect_kw("tensorscript")
        tok = self.peek()
        # "v1.0" lexes as IDENT "v1" + PUNCT "." + NUMBER "0" (the lexer's
        # NUMBER rule can't start mid-token on the "1" once "v1" has already
        # been consumed as one IDENT), so the major/minor split happens here
        # in the parser rather than in the lexer.
        if tok.kind == "IDENT" and tok.value.startswith("v") and tok.value[1:].isdigit():
            self.advance()
            major = tok.value[1:]
            minor = "0"
            if self.check_punct("."):
                self.advance()
                minor_tok = self.peek()
                if minor_tok.kind != "NUMBER":
                    raise TensorScriptSyntaxError("expected minor version number after '.'", minor_tok.line, minor_tok.col)
                self.advance()
                minor = minor_tok.value
            return f"{major}.{minor}"
        raise TensorScriptSyntaxError(f"expected version pin like 'v1.0', got {tok.value!r}", tok.line, tok.col)

    def parse_import_stmt(self):
        self.expect_kw("import")
        path_tok = self.advance()
        if path_tok.kind != "STRING":
            raise TensorScriptSyntaxError("expected string path after 'import'", path_tok.line, path_tok.col)
        self.expect_kw("as")
        alias = self.expect_ident_like()
        return {"path": path_tok.value, "alias": alias.value}

    def parse_block(self):
        if self.check_kw("dataset"):
            return self.parse_dataset_block()
        if self.check_kw("model"):
            return self.parse_model_block()
        if self.check_kw("monitor"):
            return self.parse_monitor_block()
        if self.check_kw("optimize"):
            return self.parse_optimize_block()
        if self.check_kw("pipeline"):
            return self.parse_pipeline_block()
        if self.check_kw("evaluate"):
            return self.parse_evaluate_block()
        tok = self.peek()
        raise TensorScriptSyntaxError(f"expected a top-level block, got {tok.value!r}", tok.line, tok.col)

    # ---- dataset ----

    def parse_dataset_block(self):
        self.expect_kw("dataset")
        name = self.expect_ident_like().value
        self.expect_kw("streams")
        self.expect_punct("{")
        fields = {}
        while not self.check_punct("}"):
            key_tok = self.expect_ident_like()
            key = key_tok.value
            self.expect_punct(":")
            if key == "source":
                fields["source"] = self.parse_string()
            elif key == "mix":
                fields["mix"] = self.parse_bracketed_kv_percent()
            elif key == "tokenize":
                fields["tokenize"] = self.parse_call_expr()
            elif key == "sequence_length":
                fields["sequence_length"] = self.parse_number()
            elif key == "split":
                fields["split"] = self.parse_bracketed_kv_percent()
            elif key == "auth":
                fields["auth"] = self.parse_env_ref()
            else:
                raise TensorScriptSyntaxError(f"unknown dataset field {key!r}", key_tok.line, key_tok.col)
            self.consume_optional_comma()
        self.expect_punct("}")
        return {"type": "dataset", "name": name, "fields": fields}

    # ---- model ----

    def parse_model_block(self):
        self.expect_kw("model")
        name = self.expect_ident_like().value
        self.expect_punct("{")
        fields = {}
        while not self.check_punct("}"):
            key_tok = self.expect_ident_like()
            key = key_tok.value
            self.expect_punct(":")
            if key == "base":
                fields["base"] = self.parse_base_ref()
            elif key == "quantize":
                fields["quantize"] = self.expect_ident_like().value
            elif key == "peft":
                if self.check_kw("none"):
                    self.advance()
                    fields["peft"] = None
                else:
                    fields["peft"] = self.parse_call_expr()
            elif key == "freeze":
                fields["freeze"] = self.parse_slice_expr()
            else:
                raise TensorScriptSyntaxError(f"unknown model field {key!r}", key_tok.line, key_tok.col)
            self.consume_optional_comma()
        self.expect_punct("}")
        return {"type": "model", "name": name, "fields": fields}

    def parse_base_ref(self):
        if self.peek().kind == "STRING":
            return {"kind": "literal", "value": self.advance().value}
        if self.check_kw("scratch"):
            self.advance()
            return {"kind": "scratch"}
        if self.check_kw("pipeline"):
            self.advance()
            self.expect_punct(".")
            stage = self.expect_ident_like().value
            self.expect_punct(".")
            selector = self.parse_checkpoint_selector()
            return {"kind": "pipeline_ref", "stage": stage, "selector": selector}
        tok = self.peek()
        raise TensorScriptSyntaxError("expected string, 'scratch', or pipeline.<Stage>.<selector>", tok.line, tok.col)

    def parse_checkpoint_selector(self):
        if self.check_kw("checkpoint") or self.check_kw("best") or self.check_kw("final"):
            return self.advance().value
        if self.check_kw("step"):
            self.advance()
            self.expect_punct("(")
            n = self.parse_number()
            self.expect_punct(")")
            return {"step": n}
        tok = self.peek()
        raise TensorScriptSyntaxError("expected checkpoint selector (checkpoint|best|final|step(N))", tok.line, tok.col)

    def parse_slice_expr(self):
        parts = [self.expect_ident_like().value]
        while self.check_punct("."):
            self.advance()
            parts.append(self.expect_ident_like().value)
        self.expect_punct("[")
        start = None
        if not self.check_punct(":"):
            start = self.parse_signed_number()
        self.expect_punct(":")
        end = None
        if not self.check_punct("]"):
            end = self.parse_signed_number()
        self.expect_punct("]")
        return {"path": ".".join(parts), "start": start, "end": end}

    def parse_signed_number(self):
        neg = False
        if self.check_punct("-"):
            self.advance()
            neg = True
        n = self.parse_number()
        return -n if neg else n

    # ---- monitor ----

    def parse_monitor_block(self):
        self.expect_kw("monitor")
        name = self.expect_ident_like().value
        self.expect_punct("{")
        fields = {}
        while not self.check_punct("}"):
            key_tok = self.expect_ident_like()
            key = key_tok.value
            self.expect_punct(":")
            if key == "track":
                fields["track"] = self.parse_bracketed_ident_list()
            elif key == "checkpoint_every":
                fields["checkpoint_every"] = self.parse_duration()
            elif key == "select_best_on":
                metric = self.expect_ident_like().value
                direction = self.expect_ident_like().value
                if direction not in ("minimize", "maximize"):
                    tok = self.peek()
                    raise TensorScriptSyntaxError("select_best_on requires 'minimize' or 'maximize'", tok.line, tok.col)
                fields["select_best_on"] = {"metric": metric, "direction": direction}
            elif key == "destination":
                fields["destination"] = self.parse_string()
            elif key == "auth":
                fields["auth"] = self.parse_env_ref()
            else:
                raise TensorScriptSyntaxError(f"unknown monitor field {key!r}", key_tok.line, key_tok.col)
            self.consume_optional_comma()
        self.expect_punct("}")
        return {"type": "monitor", "name": name, "fields": fields}

    # ---- optimize ----

    def parse_optimize_block(self):
        self.expect_kw("optimize")
        target = self.expect_ident_like().value
        alias = None
        if self.check_kw("as"):
            self.advance()
            alias = self.expect_ident_like().value
        self.expect_punct("{")
        fields = {}
        while not self.check_punct("}"):
            key_tok = self.expect_ident_like()
            key = key_tok.value
            self.expect_punct(":")
            if key == "using":
                fields["using"] = self.parse_dotted_ref()
            elif key == "telemetry":
                fields["telemetry"] = self.parse_dotted_ref()
            elif key == "schedule":
                fields["schedule"] = self.parse_call_expr()
            elif key == "epochs":
                fields["epochs"] = self.parse_number()
            elif key == "seed":
                fields["seed"] = self.parse_number()
            elif key == "hardware":
                fields["hardware"] = self.parse_hardware_spec()
            elif key == "guardrails":
                fields["guardrails"] = self.parse_guardrail_list()
            else:
                raise TensorScriptSyntaxError(f"unknown optimize field {key!r}", key_tok.line, key_tok.col)
            self.consume_optional_comma()
        self.expect_punct("}")
        return {"type": "optimize", "target": target, "alias": alias, "fields": fields}

    def parse_hardware_spec(self):
        self.expect_punct("{")
        spec = {}
        while not self.check_punct("}"):
            key = self.expect_ident_like().value
            self.expect_punct(":")
            if key == "nodes":
                spec["nodes"] = self.parse_number()
            elif key == "gpus_per_node":
                spec["gpus_per_node"] = self.parse_number()
            elif key == "strategy":
                spec["strategy"] = self.parse_call_expr()
            elif key == "profile":
                spec["profile"] = self.parse_string()
            else:
                tok = self.peek()
                raise TensorScriptSyntaxError(f"unknown hardware field {key!r}", tok.line, tok.col)
            self.consume_optional_comma()
        self.expect_punct("}")
        return spec

    def parse_guardrail_list(self):
        self.expect_punct("[")
        rules = []
        while not self.check_punct("]"):
            rules.append(self.parse_guardrail())
            self.consume_optional_comma()
        self.expect_punct("]")
        return rules

    def parse_guardrail(self):
        self.expect_kw("if")
        condition = self.parse_condition()
        if self.check_kw("for"):
            self.advance()
            duration = self.parse_duration()
            self.expect_punct(":")
            action = self.parse_action()
            return {"condition": condition, "for": duration, "action": action}
        if self.check_punct(":"):
            self.advance()
            action = self.parse_action()
            return {"condition": condition, "for": None, "action": action}
        if self.check_punct("{"):
            self.advance()
            statements = []
            while not self.check_punct("}"):
                statements.append(self.parse_call_expr())
                self.consume_optional_comma()
            self.expect_punct("}")
            return {"condition": condition, "for": None, "block": statements}
        tok = self.peek()
        raise TensorScriptSyntaxError("expected ':' or '{' after guardrail condition", tok.line, tok.col)

    def parse_condition(self):
        metric = self.expect_ident_like().value
        if self.check_kw("flat_lines"):
            self.advance()
            self.expect_punct("(")
            self.expect_ident_like()  # 'epsilon'
            self.expect_punct("=")  # NOTE: see grammar-bug notes below
            eps = self.parse_number()
            self.expect_punct(")")
            return {"metric": metric, "op": "flat_lines", "epsilon": eps}
        op_tok = self.advance()
        if op_tok.kind != "PUNCT" or op_tok.value not in (">", "<", ">=", "<=", "=="):
            raise TensorScriptSyntaxError("expected comparator", op_tok.line, op_tok.col)
        value = self.parse_value()
        return {"metric": metric, "op": op_tok.value, "value": value}

    def parse_action(self):
        # action = call_expr | identifier
        save = self.i
        if self.peek().kind in ("IDENT", "KEYWORD") and self.peek(1).kind == "PUNCT" and self.peek(1).value == "(":
            return self.parse_call_expr()
        tok = self.expect_ident_like()
        return tok.value

    # ---- pipeline ----

    def parse_pipeline_block(self):
        self.expect_kw("pipeline")
        name = self.expect_ident_like().value
        self.expect_punct("{")
        stages = []
        while not self.check_punct("}"):
            stages.append(self.parse_stage())
            self.consume_optional_comma()
        self.expect_punct("}")
        return {"type": "pipeline", "name": name, "stages": stages}

    def parse_stage(self):
        self.expect_kw("stage")
        name = self.expect_ident_like().value
        self.expect_punct("{")
        fields = {}
        while not self.check_punct("}"):
            key_tok = self.expect_ident_like()
            key = key_tok.value
            self.expect_punct(":")
            if key == "run":
                fields["run"] = self.parse_dotted_ref()
            elif key == "depends_on":
                fields["depends_on"] = self.expect_ident_like().value
            elif key == "inherit_weights":
                fields["inherit_weights"] = self.parse_checkpoint_selector()
            else:
                raise TensorScriptSyntaxError(f"unknown stage field {key!r}", key_tok.line, key_tok.col)
            self.consume_optional_comma()
        self.expect_punct("}")
        return {"name": name, "fields": fields}

    # ---- evaluate ----

    def parse_evaluate_block(self):
        self.expect_kw("evaluate")
        name = self.expect_ident_like().value
        self.expect_punct("{")
        fields = {}
        while not self.check_punct("}"):
            key_tok = self.expect_ident_like()
            key = key_tok.value
            self.expect_punct(":")
            if key == "on":
                fields["on"] = self.parse_dotted_ref()
            elif key == "benchmarks":
                fields["benchmarks"] = self.parse_benchmark_list()
            elif key == "after":
                if self.check_kw("every_checkpoint") or self.check_kw("final_only"):
                    fields["after"] = self.advance().value
                else:
                    fields["after"] = self.parse_duration()
            elif key == "report_to":
                fields["report_to"] = self.parse_dotted_ref()
            else:
                raise TensorScriptSyntaxError(f"unknown evaluate field {key!r}", key_tok.line, key_tok.col)
            self.consume_optional_comma()
        self.expect_punct("}")
        return {"type": "evaluate", "name": name, "fields": fields}

    def parse_benchmark_list(self):
        self.expect_punct("[")
        items = []
        while not self.check_punct("]"):
            name = self.expect_ident_like().value
            args = []
            if self.check_punct("("):
                call = self.parse_call_expr(name_already=name)
                items.append(call)
            else:
                items.append({"call": name, "args": []})
            self.consume_optional_comma()
        self.expect_punct("]")
        return items

    # ---- shared low-level rules ----

    def parse_string(self):
        tok = self.peek()
        if tok.kind != "STRING":
            raise TensorScriptSyntaxError("expected string", tok.line, tok.col)
        return self.advance().value

    def parse_number(self):
        tok = self.peek()
        if tok.kind != "NUMBER":
            raise TensorScriptSyntaxError("expected number", tok.line, tok.col)
        self.advance()
        return float(tok.value) if "." in tok.value or "e" in tok.value or "E" in tok.value else int(tok.value)

    def parse_percent(self):
        tok = self.peek()
        if tok.kind != "PERCENT":
            raise TensorScriptSyntaxError("expected percent literal (e.g. 40%)", tok.line, tok.col)
        self.advance()
        return float(tok.value)

    def parse_duration(self):
        n = self.parse_number()
        self.expect_punct(".")
        unit_tok = self.expect_ident_like()
        if unit_tok.value not in ("steps", "epochs"):
            raise TensorScriptSyntaxError("expected 'steps' or 'epochs'", unit_tok.line, unit_tok.col)
        return {"n": n, "unit": unit_tok.value}

    def parse_dotted_ref(self):
        parts = [self.expect_ident_like().value]
        while self.check_punct("."):
            self.advance()
            parts.append(self.expect_ident_like().value)
        return ".".join(parts)

    def parse_env_ref(self):
        self.expect_kw("env")
        self.expect_punct("(")
        var = self.expect_ident_like().value
        self.expect_punct(")")
        return {"env_var": var}

    def parse_call_expr(self, name_already=None):
        name = name_already if name_already is not None else self.expect_ident_like().value
        self.expect_punct("(")
        args = []
        while not self.check_punct(")"):
            args.append(self.parse_arg())
            self.consume_optional_comma()
        self.expect_punct(")")
        return {"call": name, "args": args}

    def parse_arg(self):
        # arg = [ identifier "=" ] value
        if (self.peek().kind in ("IDENT", "KEYWORD")
                and self.peek(1).kind == "PUNCT" and self.peek(1).value == "="):
            key = self.advance().value
            self.advance()  # '='
            return {"key": key, "value": self.parse_value()}
        return {"key": None, "value": self.parse_value()}

    def parse_value(self):
        tok = self.peek()
        if tok.kind == "STRING":
            return self.advance().value
        if tok.kind == "PERCENT":
            return self.parse_percent()
        if tok.kind == "NUMBER":
            return self.parse_number()
        if tok.kind == "PUNCT" and tok.value == "-":
            return self.parse_signed_number()
        if tok.kind in ("IDENT", "KEYWORD"):
            if self.peek(1).kind == "PUNCT" and self.peek(1).value == "(":
                return self.parse_call_expr()
            return self.advance().value
        raise TensorScriptSyntaxError("expected a value", tok.line, tok.col)

    def parse_bracketed_kv_percent(self):
        self.expect_punct("[")
        entries = {}
        while not self.check_punct("]"):
            key = self.expect_ident_like().value
            self.expect_punct(":")
            entries[key] = self.parse_percent()
            self.consume_optional_comma()
        self.expect_punct("]")
        return entries

    def parse_bracketed_ident_list(self):
        self.expect_punct("[")
        items = []
        while not self.check_punct("]"):
            items.append(self.expect_ident_like().value)
            self.consume_optional_comma()
        self.expect_punct("]")
        return items

    def consume_optional_comma(self):
        if self.check_punct(","):
            self.advance()


def parse(source: str):
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_file()
