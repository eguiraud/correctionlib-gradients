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


@dataclass
class Parameter:
    idx: int


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


FormulaNode: TypeAlias = Union[Literal, Variable, Parameter, "Op"]


@dataclass
class Op:
    op: BinaryOp | UnaryOp
    children: tuple[FormulaNode, ...]


class FormulaDAG:
    def __init__(self, f: schema.Formula, inputs: list[schema.Variable]):
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
            case Op(op=BinaryOp(), children=children):
                c1, c2 = children
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
                        return jnp.max(jnp.stack([ev(c1, i), ev(c2, i)]))
                    case BinaryOp.MIN:
                        return jnp.min(jnp.stack([ev(c1, i), ev(c2, i)]))
            case _:  # pragma: no cover
                msg = f"Type of formula node not recognized ({node}). This should never happen."
                raise RuntimeError(msg)

        # never reached, only here to make mypy happy
        return jax.array()  # pragma: no cover

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
            case _:  # pragma: no cover
                msg = f"Type of formula node not recognized ({ast.nodetype.name}). This should never happen."
                raise ValueError(msg)

        # never reached, just to make mypy happy
        return Literal(0.0)  # pragma: no cover
