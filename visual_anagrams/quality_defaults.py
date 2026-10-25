"""Context-specific CLI/GUI defaults. Neutral sampler options remain unchanged.

Populate only from the recorded quality pilot; explicit user choices always win.
"""

# Two matched geometric cases improved weaker-view alignment and concealment.
# This costs 6 rather than 4 model forwards per guided step at strength 0.5.
STANDARD_GEOMETRIC = dict(cross_negative='qwen')
# Two full-size matched trials: clearer far-view Vader, no colour-view regression,
# and one shared reference forward instead of two. See examples/quality-experiments.
COMPONENT_GROUPS = {
    frozenset(('low_pass','high_pass')):dict(component_guidance=True),
    frozenset(('grayscale','color')):dict(component_guidance=True),
}


def experiment_defaults(*,turbo,views,reduction,quality_method):
    if turbo:return {}
    if len(views)==2 and reduction=='mean' and quality_method=='adaptive':
        return dict(STANDARD_GEOMETRIC)
    if reduction=='sum' and quality_method=='plain':
        return dict(COMPONENT_GROUPS.get(frozenset(views),{}))
    return {}
