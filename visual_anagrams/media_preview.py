"""Bounded-memory GIF/MP4 playback for Tk; decoding never runs on the UI thread."""

from dataclasses import dataclass,field
from pathlib import Path
import math
import queue
import threading
import time

from PIL import Image


@dataclass
class _Session:
    frames:queue.Queue=field(default_factory=lambda:queue.Queue(maxsize=3))
    stop:threading.Event=field(default_factory=threading.Event)
    thread:threading.Thread|None=None


def _put(session,item):
    while not session.stop.is_set():
        try:session.frames.put(item,timeout=.05);return True
        except queue.Full:pass
    return False


def _decode(path,session):
    """Only queue PIL frames/errors. Tk objects stay on the main thread."""
    try:
        if path.suffix.lower()=='.gif':
            with Image.open(path) as source:
                while not session.stop.is_set():
                    for index in range(getattr(source,'n_frames',1)):
                        if session.stop.is_set():return
                        source.seek(index)
                        frame=source.convert('RGB')
                        frame.thumbnail((768,768),Image.Resampling.LANCZOS)
                        duration=max(10,int(source.info.get('duration',100)))
                        if not _put(session,('frame',frame,duration)):return
        else:
            import imageio_ffmpeg
            reader=imageio_ffmpeg.read_frames(str(path),input_params=['-stream_loop','-1'])
            try:
                metadata=next(reader)
                fps=float(metadata.get('fps') or 24)
                if not math.isfinite(fps) or fps<=0:fps=24
                duration=max(10,round(1000/fps));seen=False
                for pixels in reader:
                    if session.stop.is_set():return
                    frame=Image.frombytes('RGB',metadata['size'],pixels)
                    frame.thumbnail((768,768),Image.Resampling.LANCZOS)
                    seen=True
                    if not _put(session,('frame',frame,duration)):return
                if not seen:raise ValueError('The video contains no decodable frames')
                if not session.stop.is_set():raise ValueError('The video decoder stopped unexpectedly')
            finally:reader.close()
    except Exception as exc:
        if not session.stop.is_set():_put(session,('error',str(exc),0))


class AnimationPreview:
    def __init__(self,owner,on_frame,on_error,on_state):
        self.owner,self.on_frame,self.on_error,self.on_state=owner,on_frame,on_error,on_state
        self.path=None;self.session=None;self.playing=False;self.after_id=None
        self.displayed_frames=0

    def play(self,path):
        self.stop()
        self.path=Path(path);self.session=_Session();self.displayed_frames=0
        self.session.thread=threading.Thread(target=_decode,args=(self.path,self.session),daemon=True)
        self.session.thread.start()
        self.resume()

    def pause(self):
        self.playing=False
        self._cancel_tick()
        self.on_state('paused')

    def resume(self):
        if self.session is None:return
        self.playing=True;self.on_state('playing')
        if self.after_id is None:self.after_id=self.owner.after(1,self._tick)

    def stop(self):
        self._cancel_tick()
        if self.session is not None:self.session.stop.set()
        self.session=None;self.path=None;self.playing=False
        self.on_state('stopped')

    def _cancel_tick(self):
        if self.after_id is not None:
            try:self.owner.after_cancel(self.after_id)
            except Exception:pass # owner may already be inside its Destroy callback
            self.after_id=None

    def _tick(self):
        self.after_id=None
        if not self.playing or self.session is None:return
        started=time.monotonic()
        try:kind,value,duration=self.session.frames.get_nowait()
        except queue.Empty:
            self.after_id=self.owner.after(10,self._tick);return
        if kind=='error':
            self.stop();self.on_error(value);return
        self.displayed_frames+=1
        self.on_frame(value)
        delay=max(1,round(duration-(time.monotonic()-started)*1000))
        self.after_id=self.owner.after(delay,self._tick)
