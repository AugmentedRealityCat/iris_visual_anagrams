"""Real Iris depth check: add parallax to the first second of an existing MP4."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from iris_tools import main as run
    source=ROOT/'examples/depth-lab/parallax-batman-result.mp4'
    output=ROOT/'results/video-parallax-check-20261026'
    if (output/'first-second/0000/run.json').exists():
        print('Already complete',output,flush=True);return
    run(['--task','parallax_video','--input',str(source),'--video_start','0','--video_seconds','1',
        '--render_side','512','--motion','8','--depth_formats','mp4','gif',
        '--name','first-second','--save_dir',str(output)])
    record=json.loads((output/'first-second/0000/run.json').read_text())
    assert record['frames']==24 and abs(record['seconds_exported']-1)<1e-6
    print(json.dumps({key:record[key] for key in ('frames','fps','seconds_exported','seconds','peak_allocated_gib')}),flush=True)


if __name__=='__main__':main()
