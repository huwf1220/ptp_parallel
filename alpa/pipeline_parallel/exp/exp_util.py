from typing import Sequence
import os
import inspect
from jax.core import ClosedJaxpr, Jaxpr
from alpa.pipeline_parallel.computation import JaxPipelineComputation
from jax.core import JaxprPpContext, JaxprPpSettings, pp_jaxpr
from alpa.pipeline_parallel.exp.pp_util import p_jaxpr

def format_list(lst, line_length=20, indent="    "):
    """Format a list into multiple lines with a specified line length."""
    return "\n".join(
        f"{indent}{', '.join(map(str, lst[i:i + line_length]))}"
        for i in range(0, len(lst), line_length)
    )

def p_jaxpr_in_context(jaxpr, context=None, settings=None):
    """Pretty print a Jaxpr with a given context and settings."""
    context = context or JaxprPpContext()
    settings = settings or JaxprPpSettings()
    return str(pp_jaxpr(jaxpr, context, settings))

def write_jaxpr_to_file(f, jaxpr, context, settings, context_str, jaxpr_print_fn):
    """Write Jaxpr details to a file."""
    f.write("invars:\n")
    f.write(format_list(jaxpr.invars) + "\n")
    f.write("outvars:\n")
    f.write(format_list(jaxpr.outvars) + "\n")
    f.write("constvars:\n")
    f.write(format_list(jaxpr.constvars) + "\n")
    f.write(f"first_eqn invars: {jaxpr.eqns[0].invars}\n")
    f.write(f"first_eqn outvars: {jaxpr.eqns[0].outvars}\n")
    f.write("full jaxpr:\n")
    f.write(context_str)
    f.write(jaxpr_print_fn(jaxpr, context, settings))
    f.write("\n")
    f.write(f"last_eqn invars: {jaxpr.eqns[-1].invars}\n")
    f.write(f"last_eqn outvars: {jaxpr.eqns[-1].outvars}\n")

def save_jaxpr(obj, note='', current_var_name=True, context=None, settings=None):
    """Save a Jaxpr or a sequence of JaxPipelineComputations to a file."""
    frame = inspect.currentframe().f_back
    file_name = os.path.basename(frame.f_code.co_filename)
    line_no = frame.f_lineno
    local_vars = frame.f_locals
    var_name = next((name for name, value in local_vars.items() if value is obj), None)

    txt_file_name = f"{var_name}_{note}.txt"
    os.makedirs("jaxpr_output", exist_ok=True)
    txt_file_name = os.path.join("jaxpr_output", txt_file_name)

    
    context_str = "Jaxpr with NEW PP CONTEXT\n" if context is None else "Using GIVEN PP CONTEXT\n"
    jaxpr_print_fn = p_jaxpr if current_var_name else p_jaxpr_in_context

    with open(txt_file_name, "w") as f:
        if isinstance(obj, Sequence) and all(isinstance(x, JaxPipelineComputation) for x in obj):
            # Sequence[JaxPipelineComputation]
            for idx, computation in enumerate(obj):
                closed_jaxpr = computation.closed_jaxpr()
                f.write('*' * 20 + f" computation {idx} " + '*' * 20 + '\n')
                write_jaxpr_to_file(f, closed_jaxpr.jaxpr, context, settings, context_str, jaxpr_print_fn)
                f.write("\n\n")  # Two blank lines after each computation
        elif isinstance(obj, Sequence) and all(isinstance(x, ClosedJaxpr) for x in obj):
            for idx, closed_jaxpr in enumerate(obj):
                f.write('*' * 20 + f" ClosedJaxpr {idx} " + '*' * 20 + '\n')
                write_jaxpr_to_file(f, closed_jaxpr.jaxpr, context, settings, context_str, jaxpr_print_fn)
                f.write("\n\n")  # Two blank lines after each computation
        elif isinstance(obj, (ClosedJaxpr, Jaxpr)):
            # Single Jaxpr
            jaxpr = obj.jaxpr if isinstance(obj, ClosedJaxpr) else obj
            write_jaxpr_to_file(f, jaxpr, context, settings, context_str, jaxpr_print_fn)
        else:
            raise TypeError("Input object must be a Jaxpr or a Sequence[JaxPipelineComputation].")