"""Depth-driven renderers. No models or torch are imported by this module.

Depth is an artistic near/far coordinate, never a calibrated camera distance.
Stereogram geometry follows Thimbleby, Inglis & Witten (Computer, 1994).
"""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def near_map(depth):
    depth=np.asarray(depth,dtype=np.float32)
    if depth.ndim!=2 or not np.isfinite(depth).all():
        raise ValueError('Depth must be a finite two-dimensional array')
    low,high=np.percentile(depth,[2,98])
    if high-low<1e-6:return np.full_like(depth,.5)
    return np.clip((high-depth)/(high-low),0,1).astype(np.float32)


def fit_image(image, max_side=768, multiple=2):
    scale=min(1,max_side/max(image.size))
    size=tuple(max(multiple,round(side*scale/multiple)*multiple) for side in image.size)
    return image.convert('RGB').resize(size,Image.Resampling.LANCZOS)


def resize_map(array,size):
    return np.asarray(Image.fromarray(np.asarray(array,dtype=np.float32)).resize(size,Image.Resampling.BILINEAR))


def _check(image,near):
    near=np.asarray(near,dtype=np.float32)
    if near.shape!=(image.height,image.width) or not np.isfinite(near).all() or near.min()<0 or near.max()>1:
        raise ValueError('Near map must match the image and have finite values in [0, 1]')
    return near


def _fill_holes(rgb,valid):
    """Normalized local colour diffusion; this does not reconstruct hidden scenery."""
    result=rgb.copy()
    known=valid.copy()
    for radius in (1,2,4,8,16,32,64):
        if known.all():break
        # Pillow's blur filters require L/RGB, so keep matching 8-bit premultiplication.
        alpha=np.asarray(Image.fromarray((known*255).astype('uint8')).filter(ImageFilter.GaussianBlur(radius)),dtype=np.float32)/255
        premult=np.asarray(Image.fromarray(np.uint8(np.clip(result,0,255)*known[...,None])).filter(ImageFilter.GaussianBlur(radius)),dtype=np.float32)
        fresh=(~known)&(alpha>1/255)
        result[fresh]=premult[fresh]/alpha[fresh,None]
        known[fresh]=True
    if not known.all():result[~known]=result[known].mean(0) if known.any() else 127
    return np.uint8(np.clip(result,0,255))


def parallax_frame(image,near,dx=0.,dy=0.,crop=0):
    """Bilinear forward splat, near-surface z-buffer, local fill, fixed edge crop."""
    near=_check(image,near)
    if not all(math.isfinite(v) for v in (dx,dy)) or crop<0 or 2*crop>=min(image.size):
        raise ValueError('Invalid camera displacement or crop')
    rgb=np.asarray(image.convert('RGB'),dtype=np.float32)
    height,width=near.shape
    yy,xx=np.indices(near.shape,dtype=np.float32)
    tx=xx+dx*(near-.5);ty=yy+dy*(near-.5)
    ix=np.floor(tx).astype(int);iy=np.floor(ty).astype(int)
    fragments=[]
    z=np.full(height*width,-np.inf,dtype=np.float32)
    for ox,oy in ((0,0),(1,0),(0,1),(1,1)):
        x=ix+ox;y=iy+oy
        weight=(1-np.abs(tx-x))*(1-np.abs(ty-y))
        valid=(x>=0)&(x<width)&(y>=0)&(y<height)&(weight>1e-5)
        indices=(y[valid]*width+x[valid]).ravel()
        depths=near[valid]
        np.maximum.at(z,indices,depths)
        fragments.append((indices,depths,weight[valid],rgb[valid]))
    mass=np.zeros(height*width,dtype=np.float32);colour=np.zeros((height*width,3),dtype=np.float32)
    for indices,depths,weight,pixels in fragments:
        visible=depths>=z[indices]-.02
        indices=indices[visible];weight=weight[visible];pixels=pixels[visible]
        mass+=np.bincount(indices,weights=weight,minlength=height*width).astype(np.float32)
        for channel in range(3):colour[:,channel]+=np.bincount(indices,weights=weight*pixels[:,channel],minlength=height*width).astype(np.float32)
    valid=(mass>1e-6).reshape(height,width)
    pixels=(colour/np.maximum(mass[:,None],1e-6)).reshape(height,width,3)
    filled=_fill_holes(pixels,valid)
    frame=Image.fromarray(filled)
    if crop:
        frame=frame.crop((crop,crop,width-crop,height-crop)).resize((width,height),Image.Resampling.LANCZOS)
        coverage=valid[crop:-crop,crop:-crop]
    else:coverage=valid
    return frame,dict(uncovered_fraction=float(1-coverage.mean()))


def parallax_frames(image,near,frames=72,amplitude=12.):
    crop=math.ceil(amplitude*.5)+2
    for index in range(frames):
        phase=2*math.pi*index/frames
        yield parallax_frame(image,near,amplitude*math.sin(phase),amplitude*.3*math.sin(2*phase),crop)


def focus_frame(image,near,focus=.5,aperture=14.,layers=12):
    """Depth-layer defocus with premultiplied alpha and front-to-back occlusion.

    All frames render the same source pixels: no crossfade between subject images.
    """
    near=_check(image,near)
    if not 0<=focus<=1 or not math.isfinite(aperture) or aperture<0 or layers<2:
        raise ValueError('Invalid focus or aperture')
    pixels=np.asarray(image.convert('RGB'),dtype=np.float32)
    if aperture==0:return image.convert('RGB').copy()
    result=np.zeros_like(pixels);transmission=np.ones(near.shape,dtype=np.float32)
    centres=np.linspace(0,1,layers)
    labels=np.floor(near*(layers-1)+.5).astype(int)
    weights=(labels[None]==np.arange(layers)[:,None,None]).astype(np.float32)
    for level,alpha in reversed(list(zip(centres,weights))):
        radius=abs(float(level)-focus)*aperture
        a=np.asarray(Image.fromarray(np.uint8(alpha*255)).filter(ImageFilter.GaussianBlur(radius)),dtype=np.float32)/255
        c=np.asarray(Image.fromarray(np.uint8(pixels*alpha[...,None])).filter(ImageFilter.GaussianBlur(radius)),dtype=np.float32)
        result+=transmission[...,None]*c
        transmission*=1-a
    result+=transmission[...,None]*np.asarray(image.filter(ImageFilter.GaussianBlur(aperture)),dtype=np.float32)
    return Image.fromarray(np.uint8(np.clip(result,0,255)))


def focus_frames(image,near,frames=72,aperture=14.):
    for index in range(frames):
        focus=.5-.5*math.cos(2*math.pi*index/frames)
        yield focus_frame(image,near,focus,aperture),dict(focus=float(focus))


def stereogram(near,period=72,seed=0,texture=None):
    """Symmetric stereo constraints with visibility tests and rightward link chains.

    Parallel viewing: the normalized near coordinate ranges from far=0 to near=1.
    A texture colours free roots only; it cannot independently depict a full RGB image.
    """
    near=np.asarray(near,dtype=np.float32)
    if near.ndim!=2 or not np.isfinite(near).all() or near.min()<0 or near.max()>1:
        raise ValueError('Stereogram needs a finite near map in [0, 1]')
    height,width=near.shape
    if not 8<=period<=width//3:raise ValueError('Stereo period must be between 8 pixels and one third of the image width')
    rng=np.random.default_rng(seed)
    pixels=rng.integers(0,256,(height,width,3),dtype=np.uint8)
    if texture is not None:
        tile=np.asarray(texture.convert('RGB').resize((period,height),Image.Resampling.LANCZOS))
        pixels=np.uint8(.65*tile[:,np.arange(width)%period]+.35*pixels)
    eye=2*period;mu=1/3
    separation=np.floor((1-mu*near)*eye/(2-mu*near)+.5).astype(int)
    accepted=0
    for row in range(height):
        links=list(range(width));depth=near[row]
        for centre in range(width):
            gap=int(separation[row,centre])
            left=centre-(gap+((gap&1)&(row&1)))//2;right=left+gap
            if left<0 or right>=width:continue
            limit=float(depth[centre]);distance=1;visible=True
            while True:
                ray=limit+2*(2-mu*limit)*distance/(mu*eye)
                if ray>=1:break
                if centre-distance<0 or centre+distance>=width:break
                if depth[centre-distance]>=ray or depth[centre+distance]>=ray:
                    visible=False;break
                distance+=1
            if not visible:continue
            # Merge ordered equality constraints without cycles; closer conflicts
            # are retained through the shared link chain rather than overwritten.
            while links[left] not in (left,right):
                following=links[left]
                if following<right:left=following
                else:left,right=right,following
            links[left]=right;accepted+=1
        for column in range(width-1,-1,-1):
            if links[column]!=column:pixels[row,column]=pixels[row,links[column]]
    return Image.fromarray(pixels),dict(period=period,near_separation=int(separation.min()),
        far_separation=int(separation.max()),accepted_constraints=accepted,viewing='parallel / look behind the image')


def decode_stereogram(image,period):
    """Local RGB correspondence diagnostic; not a measurement of human stereo fusion."""
    pixels=np.asarray(image.convert('RGB'),dtype=np.float32)
    height,width=pixels.shape[:2];eye=2*period;mu=1/3
    gaps=np.arange(round(.8*period),period+1)
    costs=[]
    for gap in gaps:
        left=np.arange(width)-gap//2;right=left+gap
        valid=(left>=0)&(right<width)
        difference=np.full((height,width),255,dtype=np.uint8)
        difference[:,valid]=np.abs(pixels[:,left[valid]]-pixels[:,right[valid]]).mean(-1).round().astype(np.uint8)
        costs.append(np.asarray(Image.fromarray(difference).filter(ImageFilter.BoxBlur(2)),dtype=np.float32))
    separation=gaps[np.argmin(np.stack(costs),axis=0)]
    return np.clip((eye-2*separation)/(mu*(eye-separation)),0,1).astype(np.float32)


def write_video(path,frames,fps=24):
    """Backward-compatible MP4 entry point."""
    return write_animations([path],frames,fps)


def write_animations(paths,frames,fps=24):
    """Stream GIF/MP4 from one render pass; publish only completed encodings."""
    from contextlib import ExitStack
    import imageio.v2 as imageio
    from .export import _GifStreamWriter
    paths=list(dict.fromkeys(Path(path) for path in paths))
    if not paths or any(path.suffix.lower() not in ('.mp4','.gif') for path in paths):
        raise ValueError('Choose MP4, GIF, or both')
    if not math.isfinite(fps) or not 1<=fps<=60:raise ValueError('Animation FPS must be between 1 and 60')
    temporary=[path.with_name(path.stem+'.partial'+path.suffix) for path in paths]
    records=[]
    try:
        with ExitStack() as stack:
            writers=[]
            for path,temp in zip(paths,temporary):
                if path.suffix.lower()=='.gif':writer=stack.enter_context(_GifStreamWriter(temp,fps))
                else:
                    writer=imageio.get_writer(temp,format='FFMPEG',mode='I',fps=fps,codec='libx264',
                        pixelformat='yuv420p',macro_block_size=2,ffmpeg_log_level='error',
                        output_params=['-crf','22','-movflags','+faststart'])
                    # imageio's context manager skips close on exceptions. Close
                    # explicitly so FFmpeg cannot recreate a temporary after cleanup.
                    stack.callback(writer.close)
                writers.append(writer)
            for image,record in frames:
                if any(path.suffix.lower()=='.mp4' for path in paths) and (image.width%2 or image.height%2):
                    raise ValueError('Video frame dimensions must be even')
                pixels=np.asarray(image.convert('RGB'))
                for writer in writers:writer.append_data(pixels)
                records.append(record)
            if not records:raise ValueError('No animation frames were rendered')
        for temp,path in zip(temporary,paths):temp.replace(path)
    finally:
        for temp in temporary:temp.unlink(missing_ok=True)
    return records
