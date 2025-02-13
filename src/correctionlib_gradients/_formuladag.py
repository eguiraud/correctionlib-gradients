# SPDX-FileCopyrightText: 2023-present Enrico Guiraud <enrico.guiraud@pm.me>
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import dataclass
from enum import Enum, auto
from typing import TypeAlias, Union

import correctionlib.schemav2 as schema
import jax
import jax.numpy as jnp  # atan2
from correctionlib._core import Formula, FormulaAst
from correctionlib._core import Variable as CPPVariable

import correctionlib_gradients._utils as utils


@dataclass
class Literal:
    value: float


@dataclass
class Variable:
    name: str


class BinaryOp(Enum):
    EQUAL = auto()
    NOTEQUAL = auto()
    GREATER = auto()
    LESS = auto()
    GREATEREQ = auto()
    LESSEQ = auto()
    MINUS = auto()
    PLUS = auto()
    DIV = auto()
    TIMES = auto()
    POW = auto()
    ATAN2 = auto()
    MAX = auto()
    MIN = auto()


class UnaryOp(Enum):
    NEGATIVE = auto()
    LOG = auto()
    LOG10 = auto()
    EXP = auto()
    ERF = auto()
    SQRT = auto()
    ABS = auto()
    COS = auto()
    SIN = auto()
    TAN = auto()
    ACOS = auto()
    ASIN = auto()
    ATAN = auto()
    COSH = auto()
    SINH = auto()
    TANH = auto()
    ACOSH = auto()
    ASINH = auto()
    ATANH = auto()


FormulaNode: TypeAlias = Union[Literal, Variable, "Op"]


@dataclass
class Op:
    op: BinaryOp | UnaryOp
    children: tuple[FormulaNode, ...]


class FormulaDAG:
    def __init__(self, f: schema.Formula, inputs: list[schema.Variable]):
        print(f"f.json() = {f.json()}")
        print(f"inputs = {inputs}")
        cpp_formula = Formula.from_string(f.json(), [CPPVariable.from_string(v.json()) for v in inputs])
        self.input_names = [v.name for v in inputs]
        self.node: FormulaNode = self._make_node(cpp_formula.ast)

    def evaluate(self, inputs: dict[str, jax.Array]) -> jax.Array:
        res = self._eval_node(self.node, inputs)
        return res

    def _eval_node(self, node: FormulaNode, inputs: dict[str, jax.Array]) -> jax.Array:
        match node:
            case Literal(value):
                res_size = utils.get_result_size(inputs)
                if res_size == 0:
                    return jnp.array(value)
                else:
                    return jnp.repeat(value, res_size)
            case Variable(name):
                return inputs[name]
            case Op(op=BinaryOp(), children=(c1, c2)):
                ev = self._eval_node
                i = inputs
                match node.op:
                    case BinaryOp.EQUAL:
                        return (ev(c1, i) == ev(c2, i)) + 0.0
                    case BinaryOp.NOTEQUAL:
                        return (ev(c1, i) != ev(c2, i)) + 0.0
                    case BinaryOp.GREATER:
                        return (ev(c1, i) > ev(c2, i)) + 0.0
                    case BinaryOp.LESS:
                        return (ev(c1, i) < ev(c2, i)) + 0.0
                    case BinaryOp.GREATEREQ:
                        return (ev(c1, i) >= ev(c2, i)) + 0.0
                    case BinaryOp.LESSEQ:
                        return (ev(c1, i) <= ev(c2, i)) + 0.0
                    case BinaryOp.MINUS:
                        return ev(c1, i) - ev(c2, i)
                    case BinaryOp.PLUS:
                        return ev(c1, i) + ev(c2, i)
                    case BinaryOp.DIV:
                        return ev(c1, i) / ev(c2, i)
                    case BinaryOp.TIMES:
                        return ev(c1, i) * ev(c2, i)
                    case BinaryOp.POW:
                        return ev(c1, i) ** ev(c2, i)
                    case BinaryOp.ATAN2:
                        return jnp.arctan2(ev(c1, i), ev(c2, i))
                    case BinaryOp.MAX:
                        return jnp.max(jnp.stack([ev(c1, i), ev(c2, i)]), axis=0)
                    case BinaryOp.MIN:
                        return jnp.min(jnp.stack([ev(c1, i), ev(c2, i)]), axis=0)
            case Op(op=UnaryOp(), children=(child,)):
                ev = self._eval_node
                match node.op:
                    case UnaryOp.NEGATIVE:
                        return -ev(child, inputs)
                    case UnaryOp.LOG:
                        return jnp.log(ev(child, inputs))
                    case UnaryOp.LOG10:
                        return jnp.log10(ev(child, inputs))
                    case UnaryOp.EXP:
                        return jnp.exp(ev(child, inputs))
                    case UnaryOp.ERF:
                        return jax.scipy.special.erf(ev(child, inputs))
                    case UnaryOp.SQRT:
                        return jnp.sqrt(ev(child, inputs))
                    case UnaryOp.ABS:
                        return jnp.abs(ev(child, inputs))
                    case UnaryOp.COS:
                        return jnp.cos(ev(child, inputs))
                    case UnaryOp.SIN:
                        return jnp.sin(ev(child, inputs))
                    case UnaryOp.TAN:
                        return jnp.tan(ev(child, inputs))
                    case UnaryOp.ACOS:
                        return jnp.arccos(ev(child, inputs))
                    case UnaryOp.ASIN:
                        return jnp.arcsin(ev(child, inputs))
                    case UnaryOp.ATAN:
                        return jnp.arctan(ev(child, inputs))
                    case UnaryOp.COSH:
                        return jnp.cosh(ev(child, inputs))
                    case UnaryOp.SINH:
                        return jnp.sinh(ev(child, inputs))
                    case UnaryOp.TANH:
                        return jnp.tanh(ev(child, inputs))
                    case UnaryOp.ACOSH:
                        return jnp.arccosh(ev(child, inputs))
                    case UnaryOp.ASINH:
                        return jnp.arcsinh(ev(child, inputs))
                    case UnaryOp.ATANH:
                        return jnp.arctanh(ev(child, inputs))
            case _:  # pragma: no cover
                msg = f"Type of formula node not recognized ({node}). This should never happen."
                raise RuntimeError(msg)

        # never reached, only here to make mypy happy
        return jnp.array([])  # pragma: no cover

    def _make_node(self, ast: FormulaAst) -> FormulaNode:
        match ast.nodetype:
            case FormulaAst.NodeType.LITERAL:
                return Literal(ast.data)
            case FormulaAst.NodeType.VARIABLE:
                return Variable(self.input_names[ast.data])
            case FormulaAst.NodeType.BINARY:
                match ast.data:
                    # TODO reduce code duplication (code generation?)
                    case FormulaAst.BinaryOp.EQUAL:
                        return Op(
                            op=BinaryOp.EQUAL,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.NOTEQUAL:
                        return Op(
                            op=BinaryOp.NOTEQUAL,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.GREATER:
                        return Op(
                            op=BinaryOp.GREATER,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.LESS:
                        return Op(
                            op=BinaryOp.LESS,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.GREATEREQ:
                        return Op(
                            op=BinaryOp.GREATEREQ,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.LESSEQ:
                        return Op(
                            op=BinaryOp.LESSEQ,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.MINUS:
                        return Op(
                            op=BinaryOp.MINUS,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.PLUS:
                        return Op(
                            op=BinaryOp.PLUS,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.DIV:
                        return Op(
                            op=BinaryOp.DIV,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.TIMES:
                        return Op(
                            op=BinaryOp.TIMES,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.POW:
                        return Op(
                            op=BinaryOp.POW,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.ATAN2:
                        return Op(
                            op=BinaryOp.ATAN2,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.MAX:
                        return Op(
                            op=BinaryOp.MAX,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
                    case FormulaAst.BinaryOp.MIN:
                        return Op(
                            op=BinaryOp.MIN,
                            children=(self._make_node(ast.children[0]), self._make_node(ast.children[1])),
                        )
            case FormulaAst.NodeType.UNARY:
                match ast.data:
                    case FormulaAst.UnaryOp.NEGATIVE:
                        return Op(op=UnaryOp.NEGATIVE, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.LOG:
                        return Op(op=UnaryOp.LOG, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.LOG10:
                        return Op(op=UnaryOp.LOG10, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.EXP:
                        return Op(op=UnaryOp.EXP, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.ERF:
                        return Op(op=UnaryOp.ERF, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.SQRT:
                        return Op(op=UnaryOp.SQRT, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.ABS:
                        return Op(op=UnaryOp.ABS, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.COS:
                        return Op(op=UnaryOp.COS, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.SIN:
                        return Op(op=UnaryOp.SIN, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.TAN:
                        return Op(op=UnaryOp.TAN, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.ACOS:
                        return Op(op=UnaryOp.ACOS, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.ASIN:
                        return Op(op=UnaryOp.ASIN, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.ATAN:
                        return Op(op=UnaryOp.ATAN, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.COSH:
                        return Op(op=UnaryOp.COSH, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.SINH:
                        return Op(op=UnaryOp.SINH, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.TANH:
                        return Op(op=UnaryOp.TANH, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.ACOSH:
                        return Op(op=UnaryOp.ACOSH, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.ASINH:
                        return Op(op=UnaryOp.ASINH, children=(self._make_node(ast.children[0]),))
                    case FormulaAst.UnaryOp.ATANH:
                        return Op(op=UnaryOp.ATANH, children=(self._make_node(ast.children[0]),))
            case _:  # pragma: no cover
                msg = f"Type of formula node not recognized ({ast.nodetype.name}). This should never happen."
                raise ValueError(msg)

        # never reached, just to make mypy happy
        return Literal(0.0)  # pragma: no cover
