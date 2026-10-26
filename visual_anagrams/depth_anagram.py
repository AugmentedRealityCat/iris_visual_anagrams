"""Bounded RGB edits through a frozen depth estimator; an experimental reveal.

This is input optimization, not depth-conditioned diffusion or model training.
"""

import io
import math

import numpy as np
from PIL import Image,ImageOps


def correlation(left,right):
    a=np.asarray(left,dtype=np.float64).ravel().copy();b=np.asarray(right,dtype=np.float64).ravel().copy()
    a-=a.mean();b-=b.mean()
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))


def robust_variants(image):
    yield 'clean',image
    yield 'resize_75',image.resize((max(1,image.width*3//4),max(1,image.height*3//4)),Image.Resampling.BICUBIC).resize(image.size,Image.Resampling.BICUBIC)
    buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=85);buffer.seek(0)
    with Image.open(buffer) as jpeg:yield 'jpeg_85',jpeg.convert('RGB')


def optimize_depth_anagram(runner,source,target,*,steps=24,max_side=384,budget=.06,learning_rate=.025):
    import torch
    import torch.nn.functional as F
    from .optimization import gradient_memory
    from .depth_effects import fit_image

    if steps<1 or max_side<64 or max_side>512 or not math.isfinite(budget) or not 0<budget<=.25:
        raise ValueError('Use positive optimization steps, a 64–512 side and an RGB edit budget in (0, 0.25]')
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError('Learning rate must be positive and finite')
    source=fit_image(source,max_side,runner.patch)
    target=ImageOps.fit(target.convert('RGB'),source.size,method=Image.Resampling.LANCZOS)
    device=runner.device
    # Ordinary inference can populate rotary tables with inference-mode tensors.
    # Regenerate them outside inference mode before checkpointed input gradients.
    for name in ('_rope_img','_rope_txt','_rope_pix'):
        cache=getattr(runner.model.pixel,name,None)
        if cache is not None:cache.clear()
    def tensor(image):return torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].to(device,dtype=torch.float32)/255
    original=tensor(source)
    def depth(rgb):
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            result=runner.model(rgb*2-1,runner.embeddings,runner.mask).float()
        if not torch.isfinite(result).all():raise FloatingPointError('Non-finite depth during optimization')
        return result
    def norm(value):return (value-value.mean())/value.std(unbiased=False).clamp_min(1e-4)
    with torch.no_grad():
        target_depth=depth(tensor(target)).detach()
        if target_depth.std()<1e-4:raise ValueError('The target has almost constant predicted depth; choose a more structured image')
        desired=norm(target_depth)
    parameter=torch.nn.Parameter(torch.zeros(1,3,min(96,source.height),min(96,source.width),device=device))
    optimizer=torch.optim.Adam([parameter],lr=learning_rate)
    history=[];best_loss=float('inf');best=source
    # Checkpoint the DiT trunk; frozen INT8 matmuls reconstruct weights in backward.
    with torch.enable_grad(),gradient_memory(runner.model.pixel,True):
        for step in range(steps+1):
            delta=F.interpolate(parameter,size=original.shape[-2:],mode='bilinear',align_corners=False).clamp(-budget,budget)
            candidate=(original+delta).clamp(0,1)
            # Select on clean depth. Smooth alternating parameter gradients below
            # to discourage fragile high-frequency-only perturbations.
            prediction=depth(candidate)
            match=F.mse_loss(norm(prediction),desired)
            error=float(match.detach())
            if error<best_loss:
                best_loss=error
                best=Image.fromarray((candidate[0].detach().cpu().permute(1,2,0).numpy()*255).round().astype('uint8'))
            record=dict(step=step,normalized_depth_mse=error,
                correlation=correlation(prediction[0,0].detach().cpu().numpy(),target_depth[0,0].cpu().numpy()))
            history.append(record)
            print(f'Depth optimization {step}/{steps}: target correlation {record["correlation"]:.4f}',flush=True)
            if step==steps:break
            # Avoid a second model graph: occasional smoothed-gradient updates
            # retain the clean-image objective and penalize pixel-level noise.
            variation=(delta[:,:,1:]-delta[:,:,:-1]).abs().mean()+(delta[:,:,:,1:]-delta[:,:,:,:-1]).abs().mean()
            loss=match+.5*delta.square().mean()+.2*variation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Depth optimization produced a missing or non-finite RGB gradient')
            if step%2:
                with torch.no_grad():parameter.grad.copy_(F.avg_pool2d(parameter.grad,3,1,1))
            torch.nn.utils.clip_grad_norm_([parameter],1.)
            optimizer.step()
            with torch.no_grad():parameter.clamp_(-budget,budget)
    # Score the quantized pixels that are actually saved, rather than a float buffer.
    target_array=target_depth[0,0].detach().cpu().numpy()
    checks={}
    for label,image in [('initial',source),*robust_variants(best)]:
        predicted=runner(image,max_side=max_side)
        checks[label]=dict(correlation=correlation(predicted,target_array))
    difference=np.asarray(best,dtype=np.float32)/255-np.asarray(source,dtype=np.float32)/255
    mse=float(np.mean(difference**2))
    details=dict(method='bounded RGB input optimization through frozen Iris depth',
        steps=steps,budget=budget,work_size=list(source.size),learning_rate=learning_rate,
        max_rgb_change=float(np.abs(difference).max()),psnr_db=float(-10*math.log10(max(mse,1e-12))),
        history=history,robustness=checks,
        limitation='May exploit this depth estimator; a depth reveal is not proof of a human-perceived 3D shape.')
    return best,source,target,target_array,details
