# SPDX-License-Identifier: Apache-2.0
"""Contact sheets of original rendered training targets, without changing inputs."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.capture/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Complete capture required')
    args.output.mkdir(parents=True, exist_ok=True)
    for arm, rows in manifest['arms'].items():
        sheet = Image.new('RGB',(1200,432),'#f3f0e9')
        draw = ImageDraw.Draw(sheet)
        draw.text((12,8),'Rendered training targets: '+arm,fill='#101718')
        for index, row in enumerate(rows):
            image = Image.open(args.capture/row['png']).convert('RGB')
            image.thumbnail((196,111),Image.Resampling.LANCZOS)
            x, y = (index%6)*200, 34+(index//6)*132
            sheet.paste(image,(x,y))
            draw.text((x+4,y+114),f'Frame {row["frame"]:02d}',fill='#101718')
        sheet.save(args.output/(arm+'-all-frames.jpg'),quality=92)
    sheet = Image.new('RGB',(1200,520),'#f3f0e9')
    draw = ImageDraw.Draw(sheet)
    draw.text((12,8),'Original rendered training targets | final frame16 | one scene, camera x door commands',fill='#101718')
    for row, door in enumerate(('closed','interact')):
        for col, motion in enumerate(('stationary','left','right')):
            arm=motion+'_'+door
            record=manifest['arms'][arm][-1]
            image=Image.open(args.capture/record['png']).convert('RGB')
            image.thumbnail((396,224),Image.Resampling.LANCZOS)
            x,y=col*400,34+row*242
            sheet.paste(image,(x,y))
            draw.text((x+5,y+226),arm.replace('_',' '),fill='#101718')
    sheet.save(args.output/'endpoints.jpg',quality=93)


if __name__=='__main__':
    main()
