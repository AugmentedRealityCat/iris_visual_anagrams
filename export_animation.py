"""Export GIF/MP4 animations from an existing image and its run metadata."""

import argparse
from pathlib import Path

from visual_anagrams.export import add_export_args, export_saved, options_from_args


def main(argv=None):
    parser = add_export_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--im_path", required=True)
    parser.add_argument("--metadata_path")
    args = parser.parse_args(argv)
    try:
        options = options_from_args(args)
        image = Path(args.im_path)
        metadata = Path(args.metadata_path) if args.metadata_path else image.parent.parent / "metadata.pkl"
        export_saved(image, metadata, args.animation_formats or ["mp4"], options)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
