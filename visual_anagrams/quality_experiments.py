"""Inference-only flow experiments. No model gradients or training are required."""

import torch
import torch.nn.functional as F


def scheduled_guidance(scale, time, options):
    if not 0 < time < 1:
        return 1.
    if options.guidance_schedule == 'constant':
        return scale
    def smooth(value):
        value=max(0.,min(1.,value))
        return value*value*(3-2*value)
    window=smooth((time-options.guidance_low)/options.guidance_ramp)*smooth((options.guidance_high-time)/options.guidance_ramp)
    return 1+(scale-1)*window


def project_conflicts(directions, strength):
    """Damped, symmetric PCGrad-inspired projection of conditional directions only."""
    left,right=directions
    dot=(left.float()*right.float()).sum()
    norm_l=left.float().square().sum()
    norm_r=right.float().square().sum()
    cosine=float(dot/(norm_l*norm_r).sqrt().clamp_min(1e-12))
    if dot >= 0 or strength == 0:
        return directions,dict(cosine=cosine,projected=False)
    return [left-strength*dot/norm_r.clamp_min(1e-12)*right,
            right-strength*dot/norm_l.clamp_min(1e-12)*left],dict(cosine=cosine,projected=True)


def _gaussian(image, sigma, size):
    coords=torch.arange(size,dtype=image.dtype,device=image.device)-(size-1)/2
    kernel=torch.exp(-.5*(coords/sigma)**2);kernel/=kernel.sum()
    radius=size//2
    image=F.conv2d(F.pad(image[None],(radius,radius,0,0),mode='reflect'),
        kernel.reshape(1,1,1,-1).repeat(3,1,1,1),groups=3)
    return F.conv2d(F.pad(image,(0,0,radius,radius),mode='reflect'),
        kernel.reshape(1,1,-1,1).repeat(3,1,1,1),groups=3)[0]


def component_previews(preview, views):
    """Score observable blurred/grayscale/full-detail views, not signed residuals."""
    from PIL import Image
    pixels=preview.detach().cpu().float()
    images=[]
    for view in views:
        name=view.component_name
        if name == 'low_pass':
            factor=pixels.shape[-2]//64
            shown=_gaussian(pixels,view.sigma*factor,view.kernel_size*factor+((factor+1)%2))
        elif name == 'grayscale':shown=pixels.mean(0,keepdim=True).expand_as(pixels)
        elif name == 'motion':
            factor=pixels.shape[-2]//64
            size=view.size*factor+((factor+1)%2)
            padded=F.pad(pixels,(size//2,)*4)
            shown=torch.zeros_like(pixels)
            height,width=pixels.shape[-2:]
            for offset in range(size):shown+=padded[:,offset:offset+height,offset:offset+width]
            shown/=size
        else:shown=pixels
        array=shown.clamp(-1,1).add(1).mul(127.5).round().byte().permute(1,2,0).numpy()
        images.append(Image.fromarray(array))
    return images


class ComponentGuidance:
    def __init__(self, views, scorer, freeze_below=0.):
        if scorer is None:raise ValueError('Component guidance requires a CPU preview scorer')
        self.views,self.scorer,self.freeze_below=views,scorer,freeze_below
        self.gains=torch.ones(len(views),dtype=torch.float32)
        self.history=[]

    def update(self, preview, time, step):
        scores=None
        if self.freeze_below < time < .95 and step%2 == 0:
            matrix=torch.as_tensor(self.scorer.score_images(component_previews(preview[0],self.views)))
            if matrix.shape != (len(self.views),len(self.views)) or not torch.isfinite(matrix).all():
                raise FloatingPointError('Component guidance needs a finite square preview score matrix')
            scores=matrix.diag()
            desired=(torch.softmax(-10*scores,dim=0)*len(self.views)).clamp(.5,1.5)
            self.gains=.5*self.gains+.5*desired
        self.history.append(dict(step=step,time=time,scores=scores.tolist() if scores is not None else None,
            guidance_gains=self.gains.tolist()))


class ExperimentalFlow:
    def __init__(self, options, views, reduction, guidance_scale, component_scorer=None):
        self.options,self.views,self.reduction,self.guidance_scale=options,views,reduction,guidance_scale
        self.component=ComponentGuidance(views,component_scorer,options.freeze_below) if options.component_guidance else None
        self.history=[]

    def predict(self, model, image, t_model, positive, negative, view_negative, mixer, step, num_timesteps):
        time=float(t_model[0])/num_timesteps
        scale=scheduled_guidance(self.guidance_scale,time,self.options)
        dtype=next(model.parameters()).dtype
        def forward(x, encoding, index):
            return model(x.to(dtype),t_model,encoding.embeddings[index:index+1],
                y_mask=encoding.mask[index:index+1]).x.float()
        if self.component is not None:
            # Preserve one shared reference: u + sum_i s_i P_i(c_i-u), sum_i P_i = I.
            shared=forward(image,negative,0) if scale != 1 else torch.zeros_like(image)
            directions=[]
            for i,view in enumerate(self.views):
                conditional=forward(image,positive,i)
                directions.append(view.inverse_view((conditional-shared)[0].clone())[None])
            def combine():
                scales=[1+(scale-1)*float(g) for g in self.component.gains]
                return shared+sum(s*d for s,d in zip(scales,directions)),scales
            combined,scales=combine()
            self.component.update(image-time*combined,time,step)
            combined,scales=combine()
            record=dict(step=step,time=time,guidance_scale=scale,component_scales=scales,
                component_balance=self.component.history[-1])
        else:
            indices=[step%len(self.views)] if self.reduction == 'alternate' else range(len(self.views))
            aligned,bases,directions=[],[],[]
            project=self.options.conflict_projection and .05 < time < .95
            for i in indices:
                view=self.views[i]
                viewed=view.view(image[0].clone())[None]
                conditional=forward(viewed,positive,i)
                needs_shared=scale != 1 and (self.options.cross_negative == 'off' or self.options.cross_strength != 1)
                base=forward(viewed,negative,0) if needs_shared or project else torch.zeros_like(conditional)
                reference=base
                if self.options.cross_negative != 'off' and self.options.cross_strength and scale != 1:
                    other=forward(viewed,view_negative,i)
                    reference=base.lerp(other,self.options.cross_strength)
                guided=reference+scale*(conditional-reference) if scale != 1 else conditional
                aligned.append(view.inverse_view(guided[0].clone())[None])
                if project:
                    bases.append(view.inverse_view(base[0].clone())[None])
                    directions.append(view.inverse_view((guided-base)[0].clone())[None])
            conflict=None
            if project:
                directions,conflict=project_conflicts(directions,self.options.conflict_strength)
                aligned=[base+direction for base,direction in zip(bases,directions)]
            combined=(mixer(image,aligned,time,step) if mixer else
                      torch.stack(aligned).mean(0) if self.reduction == 'mean' else torch.stack(aligned).sum(0))
            record=dict(step=step,time=time,guidance_scale=scale,conflict=conflict)
        if not torch.isfinite(combined).all():raise FloatingPointError('Experimental flow produced non-finite predictions')
        self.history.append(record)
        return combined

    def diagnostics(self):
        return dict(options=self.options.as_dict(),steps=self.history)
