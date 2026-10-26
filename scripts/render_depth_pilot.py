"""Render final depth-pilot examples using cached depth; no model weights are loaded."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    from iris_tools import main as render
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'results/depth-lab-20261026')
    args=parser.parse_args();out=args.input
    for name,task,extra in [
        ('parallax-batman','parallax',['--motion','18']),
        ('stereogram-batman','stereogram',['--texture',str(out/'joker.png')]),
        ('stereogram-dots','stereogram',[]),
        ('focus-hybrid','focus',['--aperture','24']),
    ]:
        source=out/('stereogram-batman' if name=='stereogram-dots' else name)
        if (out/'render-final'/name/'0000/run.json').exists():continue
        render(['--task',task,'--input',str(source/'input.png'),'--depth_map',str(source/'depth.npy'),
            '--save_dir',str(out/'render-final'),'--name',name,'--render_side','512','--device','cpu',*extra])


if __name__=='__main__':main()
