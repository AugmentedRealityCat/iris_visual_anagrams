"""Quality experiments and context defaults shared by the CLI, GUI and samplers."""

from dataclasses import dataclass, asdict
import math

NEGATIVE_MODES = {'Off': 'off', 'Other subject (automatic)': 'subjects',
                  'Qwen subject extraction': 'qwen', 'Manual per-view negatives': 'manual'}
GUI_DEFAULTS = dict(exp_contrastive='no',exp_negative='Off',exp_cross_strength='0.5',exp_manual_negatives='',
    exp_component='no',exp_schedule='constant',exp_low='0.1',exp_high='0.9',exp_ramp='0.05',
    exp_conflict='no',exp_conflict_strength='0.5',exp_freeze='0')


@dataclass
class ExperimentOptions:
    contrastive: bool = False
    cross_negative: str = 'off'
    cross_strength: float = .5
    component_guidance: bool = False
    guidance_schedule: str = 'constant'
    guidance_low: float = .1
    guidance_high: float = .9
    guidance_ramp: float = .05
    conflict_projection: bool = False
    conflict_strength: float = .5
    freeze_below: float = 0.

    @property
    def enabled(self):
        return (self.contrastive or self.cross_negative != 'off' or self.component_guidance
                or self.guidance_schedule != 'constant' or self.conflict_projection or self.freeze_below > 0)

    def validate(self, views, *, reduction, turbo=False, quality_method='plain'):
        if self.cross_negative not in NEGATIVE_MODES.values():
            raise ValueError('Unknown cross-view negative prompt mode')
        if self.guidance_schedule not in ('constant', 'middle'):
            raise ValueError('Guidance schedule must be constant or middle')
        numbers = [self.cross_strength, self.guidance_low, self.guidance_high, self.guidance_ramp,
                   self.conflict_strength, self.freeze_below]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError('Experimental quality parameters must be finite')
        if not 0 <= self.cross_strength <= 1 or not 0 <= self.conflict_strength <= 1:
            raise ValueError('Cross-negative and conflict strengths must be between 0 and 1')
        if not 0 <= self.guidance_low < self.guidance_high <= 1:
            raise ValueError('Guidance noise interval must satisfy 0 <= low < high <= 1')
        if not 0 < self.guidance_ramp <= (self.guidance_high-self.guidance_low)/2:
            raise ValueError('Guidance ramp must be positive and at most half the interval width')
        if not 0 <= self.freeze_below < 1:
            raise ValueError('Balance freeze noise level must be in [0, 1)')
        if not self.enabled:
            return self
        if quality_method not in ('plain', 'adaptive'):
            raise ValueError('New quality experiments use Classic or Adaptive, not Anagram-MTL/SyncVC')
        geometric = reduction == 'mean'
        if self.contrastive and (len(views) != 2 or not geometric or quality_method != 'adaptive'):
            raise ValueError('Contrastive balancing requires two geometric views and Adaptive balancing')
        if self.cross_negative != 'off' and (len(views) != 2 or not geometric):
            raise ValueError('Cross-view negatives require two geometric views with Mean/Auto combining')
        if self.conflict_projection and (len(views) != 2 or not geometric):
            raise ValueError('Conflict projection requires two geometric views with Mean/Auto combining')
        if self.component_guidance and (reduction != 'sum' or quality_method != 'plain'):
            raise ValueError('Component guidance requires a complete component group, Sum/Auto and Classic balancing')
        if self.component_guidance:
            from .composition import resolve_reduction
            # Explicit Sum on arbitrary geometric views is not a component partition.
            names = [v if isinstance(v, str) else getattr(v, 'component_name', None) for v in views]
            if not all(n in ('low_pass','high_pass','grayscale','color','motion','motion_res') for n in names):
                raise ValueError('Component guidance supports low/high frequencies, grayscale/colour or motion/residual pairs')
            resolve_reduction('auto', views)
        if turbo and (self.cross_negative != 'off' or self.component_guidance
                      or self.guidance_schedule != 'constant' or self.conflict_projection):
            raise ValueError('Cross-view negatives, component guidance, scheduled CFG and conflict projection require Standard Iris')
        if self.freeze_below and quality_method != 'adaptive' and not self.component_guidance:
            raise ValueError('Balance freeze requires Adaptive balancing or component guidance')
        return self

    def as_dict(self):
        return asdict(self)

    def cli_args(self, *, explicit=False):
        args=[]
        if self.contrastive:args+=['--contrastive_balancing']
        elif explicit:args+=['--no_contrastive_balancing']
        if self.cross_negative != 'off' or explicit:args+=['--cross_negative',self.cross_negative,'--cross_negative_strength',str(self.cross_strength)]
        if self.component_guidance:args+=['--component_guidance']
        elif explicit:args+=['--no_component_guidance']
        if self.guidance_schedule != 'constant':
            args+=['--guidance_schedule',self.guidance_schedule,'--guidance_low',str(self.guidance_low),
                '--guidance_high',str(self.guidance_high),'--guidance_ramp',str(self.guidance_ramp)]
        elif explicit:args+=['--guidance_schedule','constant']
        if self.conflict_projection:args+=['--conflict_projection','--conflict_strength',str(self.conflict_strength)]
        elif explicit:args+=['--no_conflict_projection']
        if self.freeze_below or explicit:args+=['--balance_freeze_below',str(self.freeze_below)]
        return args


def add_experiment_args(parser):
    def boolean(name):
        group=parser.add_mutually_exclusive_group()
        group.add_argument('--'+name,dest=name,action='store_true')
        group.add_argument('--no_'+name,dest=name,action='store_false')
        parser.set_defaults(**{name:None})
    boolean('contrastive_balancing')
    parser.add_argument('--cross_negative', choices=list(NEGATIVE_MODES.values()), default=None)
    parser.add_argument('--view_negative_prompts', nargs='+', help='One manual negative per view; omit the shared style')
    parser.add_argument('--negative_fallback', action='store_true',
        help='If automatic Qwen subject extraction fails, log the reason and use shared negatives only')
    parser.add_argument('--cross_negative_strength', type=float, default=None, help='Blend shared negative with other-subject negative: 0..1')
    boolean('component_guidance')
    parser.add_argument('--guidance_schedule', choices=['constant','middle'], default=None)
    parser.add_argument('--guidance_low', type=float, default=None, help='Lower flow-noise bound for scheduled CFG')
    parser.add_argument('--guidance_high', type=float, default=None)
    parser.add_argument('--guidance_ramp', type=float, default=None)
    boolean('conflict_projection')
    parser.add_argument('--conflict_strength', type=float, default=None)
    parser.add_argument('--balance_freeze_below', type=float, default=None)
    return parser


def options_from_args(args):
    from .composition import resolve_reduction
    from .method_options import default_quality_method
    reduction=resolve_reduction(args.reduction,args.views,args.view_args)
    method=args.quality_method or default_quality_method(turbo=args.turbo,views=args.views,reduction=reduction)
    values=context_options(turbo=args.turbo,views=args.views,reduction=reduction,quality_method=method).as_dict()
    for field,name in ARG_FIELDS.items():
        value=getattr(args,name)
        if value is not None:values[field]=value
    return ExperimentOptions(**values)


ARG_FIELDS=dict(contrastive='contrastive_balancing',cross_negative='cross_negative',cross_strength='cross_negative_strength',
    component_guidance='component_guidance',guidance_schedule='guidance_schedule',guidance_low='guidance_low',
    guidance_high='guidance_high',guidance_ramp='guidance_ramp',conflict_projection='conflict_projection',
    conflict_strength='conflict_strength',freeze_below='balance_freeze_below')


def resolve_args(args, options):
    for field,name in ARG_FIELDS.items():setattr(args,name,getattr(options,field))


def context_options(**context):
    from .quality_defaults import experiment_defaults
    return ExperimentOptions(**experiment_defaults(**context))


def gui_values(options):
    return dict(GUI_DEFAULTS,exp_contrastive='yes' if options.contrastive else 'no',
        exp_negative=next(label for label,value in NEGATIVE_MODES.items() if value==options.cross_negative),
        exp_cross_strength=str(options.cross_strength),exp_component='yes' if options.component_guidance else 'no',
        exp_schedule=options.guidance_schedule,exp_low=str(options.guidance_low),exp_high=str(options.guidance_high),
        exp_ramp=str(options.guidance_ramp),exp_conflict='yes' if options.conflict_projection else 'no',
        exp_conflict_strength=str(options.conflict_strength),exp_freeze=format(options.freeze_below,'g'))


def options_from_gui(settings, defaults=None):
    values=gui_values(defaults or ExperimentOptions())
    values.update({k:v for k,v in settings.items() if k in GUI_DEFAULTS})
    if values['exp_negative'] not in NEGATIVE_MODES:raise ValueError('Choose a valid cross-negative mode')
    try:
        return ExperimentOptions(values['exp_contrastive']=='yes',NEGATIVE_MODES[values['exp_negative']],
            float(values['exp_cross_strength']),values['exp_component']=='yes',values['exp_schedule'],
            float(values['exp_low']),float(values['exp_high']),float(values['exp_ramp']),
            values['exp_conflict']=='yes',float(values['exp_conflict_strength']),float(values['exp_freeze']))
    except ValueError as exc:raise ValueError('Experimental quality parameters must be numbers') from exc
