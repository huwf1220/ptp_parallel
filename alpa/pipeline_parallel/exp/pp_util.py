from typing import (Any, Callable, ClassVar, DefaultDict, Dict,
                    Generator, Hashable, Iterable, Iterator, List,
                    NamedTuple, Optional, Sequence, Set, Tuple, Type,
                    Union, cast)
import jax._src.pretty_printer as pp
from jax.core import ClosedJaxpr, Jaxpr, JaxprPpContext, JaxprPpSettings
from jax._src import source_info_util

def pp_kv_pair(k:str, v: Any, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if type(v) is tuple and all(isinstance(j, (Jaxpr, ClosedJaxpr)) for j in v):
    pp_v = pp_jaxprs(v, context, settings)
  elif isinstance(v, Jaxpr):
    pp_v = pp_jaxpr(v, context, settings)
  elif isinstance(v, ClosedJaxpr):
    pp_v = pp_jaxpr(v.jaxpr, context, settings)
  else:
    pp_v = pp.text(str(v))
  return pp.text(f'{k}=') + pp_v

def pp_kv_pairs(kv_pairs, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  if not kv_pairs:
    return pp.nil()
  return pp.group(
    pp.nest(2, pp.concat([
      pp.text("["),  pp.brk(""),
      pp.join(pp.brk(), [pp_kv_pair(k, v, context, settings) for k, v in kv_pairs])
    ]))
    + pp.brk("") + pp.text("]")
  )

def pp_eqn(eqn, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  annotation = (source_info_util.summarize(eqn.source_info)
                if settings.source_info else None)
  name_stack_annotation = f'[{eqn.source_info.name_stack}]' if settings.name_stack else None
  lhs = pp.text(str(eqn.outvars))
  rhs = [pp.text(eqn.primitive.name, annotation=name_stack_annotation),
         pp_kv_pairs(sorted(eqn.params.items()), context, settings),
         pp.text(" ") + pp.text(str(eqn.invars))]
  return pp.concat([lhs, pp.text(" = ", annotation=annotation), *rhs])

def pp_eqns(eqns, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  return pp.join(
    pp.brk("; "),
    [pp_eqn(e, context, settings) for e in eqns])

def pp_jaxpr_skeleton(jaxpr, eqns_fn, context: JaxprPpContext,
                      settings: JaxprPpSettings) -> pp.Doc:
  constvars =pp.text(str(jaxpr.constvars))
  invars = pp.text(str(jaxpr.invars))
  eqns = eqns_fn()
  outvars = pp.concat([
    pp.text("("), pp.text(str(jaxpr.outvars)),
    pp.text(")" if len(jaxpr.outvars) != 1 else ",)")])
  return pp.group(pp.nest(2, pp.concat([
    pp.text("{ "), pp.keyword(pp.text("lambda ")),
    constvars, pp.text("; "), invars,
    pp.text(". "), pp.keyword(pp.text("let")),
    pp.nest(2, pp.brk() + eqns), pp.brk(),
    pp.keyword(pp.text("in ")), outvars
  ])) + pp.text(" }"))

def pp_jaxpr(jaxpr, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  eqns_fn = lambda: pp_eqns(jaxpr.eqns, context, settings)
  return pp_jaxpr_skeleton(jaxpr, eqns_fn, context, settings)

def pp_jaxprs(jaxprs, context: JaxprPpContext, settings: JaxprPpSettings) -> pp.Doc:
  jaxprs = [j.jaxpr if isinstance(j, ClosedJaxpr) else j for j in jaxprs]
  return pp.group(pp.nest(2, pp.concat([
      pp.text('('), pp.brk(""),
      pp.join(pp.brk(), map(lambda x: pp_jaxpr(x, context, settings), jaxprs))]
    )) + pp.brk("") + pp.text(')')
  )

# Forcely print vars name instead of regenerate name with pp_var under new context.
def p_jaxpr(jaxpr, context, settings):
  context = context or JaxprPpContext()
  settings = settings or JaxprPpSettings()
  return str(pp_jaxpr(jaxpr, context, settings))