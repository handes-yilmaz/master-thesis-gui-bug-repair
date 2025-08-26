#!/usr/bin/env python
import argparse
from PIL import Image, ImageChops, ImageStat

def main(before, after, out):
    a = Image.open(before).convert('RGBA')
    b = Image.open(after).convert('RGBA')
    diff = ImageChops.difference(a, b)
    stat = ImageStat.Stat(diff)
    # simple metric: mean per channel
    mean = [round(x,2) for x in stat.mean]
    diff.save(out)
    print({'mean_diff': mean, 'out': out})

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('before')
    ap.add_argument('after')
    ap.add_argument('out')
    args = ap.parse_args()
    main(args.before, args.after, args.out)
