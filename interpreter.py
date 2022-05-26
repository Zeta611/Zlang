from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

"""
x=5;
y = (x=3;x+2)+x;
show y;
"""


@dataclass(frozen=True)
class Sum:
    left: "Exp"
    right: "Exp"


@dataclass(frozen=True)
class Difference:
    left: "Exp"
    right: "Exp"


@dataclass(frozen=True)
class Negation:
    exp: "Exp"


@dataclass(frozen=True)
class Product:
    left: "Exp"
    right: "Exp"


@dataclass(frozen=True)
class Quotient:
    left: "Exp"
    right: "Exp"


@dataclass(frozen=True)
class Int:
    val: int


@dataclass
class Assign:
    name: str
    val: "Exp"
    body: "Exp"


@dataclass
class Var:
    name: str


Exp = Int | Var | Assign | Sum | Difference | Negation | Product | Quotient
Env = dict[str, int]


def eval(exp: Exp, env: Env) -> int:
    match exp:
        case Sum(left, right):
            return eval(left, env) + eval(right, env)
        case Difference(left, right):
            return eval(left, env) - eval(right, env)
        case Negation(exp):
            return -eval(exp, env)
        case Product(left, right):
            return eval(left, env) * eval(right, env)
        case Quotient(left, right):
            return eval(left, env) // eval(right, env)
        case Int(val):
            return val
        case Assign(name, val, body):
            env = env.copy()
            env[name] = eval(val, env)
            return eval(body, env)
        case Var(name):
            try:
                return env[name]
            except KeyError:
                raise RuntimeError("Unbound variable", name) from None
        case _:
            raise RuntimeError("Unknown type", type(exp))


@dataclass(frozen=True)
class VarToken:
    name: str


@dataclass(frozen=True)
class IntToken:
    val: int


@dataclass(frozen=True)
class OpToken:
    val: str  # "+", "-", "*", "/"


@dataclass(frozen=True)
class ParToken:
    is_left: bool


@dataclass(frozen=True)
class EndToken:
    pass


Token = VarToken | IntToken | OpToken | ParToken | EndToken


def lexer(code: str, prev: Token | None = None) -> list[Token]:
    if code == "":
        # Base case: If no code is left to lex, return an empty list
        return []

    is_subtraction = True

    if code[0] == "-":
        match prev:
            case None:
                is_subtraction = False
                code = code[1:]
            case OpToken(val):
                is_subtraction = False
                code = code[1:]
            case ParToken(is_left=True):
                is_subtraction = False
                code = code[1:]
            case EndToken():
                is_subtraction = False
                code = code[1:]

    if "0" <= code[0] <= "9":
        # Number; consume all following digits
        i = 0
        integer = ""

        while len(code) > i and "0" <= code[i] <= "9":
            integer += code[i]
            i += 1
        # 523
        #    ^code[i:]

        # -1
        # 1+-1
        # (-
        # ^^^^^
        # [IntToken(-523), OpToken("+"), IntToken(1234)]

        if is_subtraction:
            token = IntToken(int(integer))
        else:
            token = IntToken(-int(integer))

        rest = lexer(code[i:], token)
        # rest: [IntToken(523), OpToken("+")]
        rest.append(token)
        # rest: [IntToken(523), OpToken("+"), IntToken(1234)]
        return rest
    elif code[0].isalpha() or code[0] == "_":
        i = 0
        var_name = ""

        while len(code) > i and (code[i].isalnum() or code[i] == "_"):
            var_name += code[i]
            i += 1

        rest = lexer(code[i:], VarToken(var_name))
        rest.append(VarToken(var_name))

        if not is_subtraction:
            rest.append(OpToken("*"))
            rest.append(IntToken(-1))

        return rest
    elif code[0] in "+-*/=":
        # Operator
        rest = lexer(code[1:], prev=OpToken(code[0]))
        # +523
        #  ^code[1:]
        # rest: [IntToken(523)]
        rest.append(OpToken(code[0]))
        # rest: [IntToken(523), OpToken("+")]
        return rest
    elif code[0] in "()":
        # Parenthesis
        rest = lexer(code[1:], prev=ParToken(code[0] == "("))
        rest.append(ParToken(code[0] == "("))

        if not is_subtraction:
            rest.append(OpToken("*"))
            rest.append(IntToken(-1))

        return rest
    elif code[0] in " \t\n\r":
        # Whitespace
        return lexer(code[1:], prev)
    elif code[0] == ";":
        rest = lexer(code[1:], prev=EndToken())
        rest.append(EndToken())
        return rest
    else:
        # Unknown
        raise SyntaxError(f"Unknown token {code[0]}")


ExpInit = Callable[[Exp, Exp], Exp]
_op_lookup: dict[str, ExpInit] = {
    "+": Sum,
    "-": Difference,
    "*": Product,
    "/": Quotient,
}


def unknown_op():
    raise ValueError("Unknown operator")


op_lookup: defaultdict[str, ExpInit] = defaultdict(unknown_op)
op_lookup.update(_op_lookup)

op_precedence = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
}


def parser(tokens: list[Token], nested_level: int = 0) -> tuple[Exp, int]:
    pos = 0
    op_stk: list[str] = []
    exp_stk: list[Exp] = []

    def handle_top(op: str | None = None):
        top = op_stk.pop()
        rhs = exp_stk.pop()
        lhs = exp_stk.pop()
        exp_stk.append(op_lookup[top](lhs, rhs))
        if op:
            op_stk.append(op)

    while len(tokens) > pos:
        match tokens[pos]:
            case IntToken(val):
                exp_stk.append(Int(val))
                pos += 1
            case OpToken("="):
                if not exp_stk or not isinstance(var := exp_stk.pop(), Var):
                    raise SyntaxError("Unexpected assignment")

                pos += 1
                val, k = parser(tokens[pos:], nested_level)
                pos += k
                body, k = parser(tokens[pos:], nested_level)
                exp_stk.append(Assign(var.name, val, body))
                pos += k
                while op_stk:
                    handle_top()
                return exp_stk[0], pos
            case OpToken(op):
                if not op_stk:
                    op_stk.append(op)
                else:
                    if op_precedence[op] <= op_precedence[op_stk[-1]]:
                        handle_top(op)
                    else:
                        op_stk.append(op)

                pos += 1
            case ParToken(is_left=True):  # (
                # Nested left parenthesis not followed by an operator
                pos += 1
                exp, k = parser(tokens[pos:], nested_level + 1)
                exp_stk.append(exp)
                pos += k
                # assert tokens[pos - 1] == ParToken(is_left=False)
            case ParToken(is_left=False):  # )
                if nested_level == 0:
                    raise SyntaxError("Unbalanced parentheses")
                pos += 1
                while op_stk:
                    handle_top()
                return exp_stk[0], pos
            case VarToken(name):
                exp_stk.append(Var(name))
                pos += 1
            case EndToken():
                pos += 1
                while op_stk:
                    handle_top()
                return exp_stk[0], pos
            case _:
                raise SyntaxError("Syntax Error")

    while op_stk:
        handle_top()
    return exp_stk[0], pos


env = {}

while True:
    try:
        user_code = input(">>> ")
        print(eval(parser(lexer(user_code)[::-1])[0], env.copy()))
        #  tokens = lexer(user_code)[::-1]
        #  ast = parser(tokens)[0]
        #  result = eval(ast, env.copy())
        #  print(f"{tokens=}\n{ast=}\n{result=}")
    except EOFError:
        print("Bye!")
        break
