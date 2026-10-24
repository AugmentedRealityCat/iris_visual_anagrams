"""Choose a valid prediction combination without importing model dependencies."""

from collections import Counter
import math


_GROUPS = (
    frozenset(("low_pass", "high_pass")),
    frozenset(("triple_low_pass", "triple_medium_pass", "triple_high_pass")),
    frozenset(("grayscale", "color")),
    frozenset(("motion", "motion_res")),
    frozenset(("scale",)),
)
_COMPONENTS = frozenset().union(*_GROUPS)


def resolve_reduction(reduction, views, view_args=None):
    """Use mean for geometric views and sum for complete component partitions.

    Accept view names (GUI/CLI validation) or instances (sampler validation).
    Component parameters must describe the same decomposition on every branch.
    """
    if reduction not in ("auto", "mean", "sum", "alternate"):
        raise ValueError("Reduction must be auto, mean, sum, or alternate")
    views = list(views)
    if view_args is not None and len(view_args) != len(views):
        raise ValueError("Expected one parameter per view")
    names = [v if isinstance(v, str) else getattr(v, "component_name", None) for v in views]
    components = [name for name in names if name in _COMPONENTS]
    if not components:
        return "mean" if reduction == "auto" else reduction
    group = next((g for g in _GROUPS if set(components) <= g), None)
    if group is None or len(components) != len(views) or set(components) != group:
        raise ValueError("Use one complete component group: low_pass + high_pass, grayscale + color, "
                         "all three triple bands, motion + motion_res, or scale weights summing to 1. "
                         "Do not mix these with geometric views.")
    if group != frozenset(("scale",)) and any(n != 1 for n in Counter(components).values()):
        raise ValueError("Use each component exactly once")

    parameters = []
    for index, (name, view) in enumerate(zip(names, views)):
        arg = None if view_args is None else view_args[index]
        if isinstance(view, str):
            if name in ("low_pass", "high_pass"):
                params = (2.0 if arg is None else float(arg), 33)
            elif name.startswith("triple_"):
                params = (1.0, 2.0, 25)
            elif name in ("motion", "motion_res"):
                params = (7,)
            elif name == "scale":
                params = (0.5 if arg is None else float(arg),)
            else:
                params = ()
        elif name in ("low_pass", "high_pass"):
            params = (view.sigma, view.kernel_size)
        elif name.startswith("triple_"):
            params = (view.sigma_1, view.sigma_2, view.kernel_size)
        elif name in ("motion", "motion_res"):
            params = (view.size,)
        elif name == "scale":
            params = (view.scale,)
        else:
            params = ()
        if not all(math.isfinite(p) for p in params):
            raise ValueError("Component parameters must be finite")
        if name != "scale" and any(p <= 0 for p in params):
            raise ValueError("Filter sizes and sigmas must be positive")
        parameters.append(params)
    if names[0] == "scale":
        if not math.isclose(math.fsum(p[0] for p in parameters), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Scale component weights must sum to 1")
    elif any(p != parameters[0] for p in parameters[1:]):
        raise ValueError("Paired components must use matching filter parameters (including blur sigma)")
    if reduction not in ("auto", "sum"):
        raise ValueError("Component methods require sum (or auto), not mean or alternate")
    return "sum"
