import argparse
import os

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000
from tqdm import tqdm


def _bgr2lab(path):
    """Read image with cv2 (BGR uint8) and convert to Lab uint8 (cv2 scale)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f'cv2 failed to read: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)


def _deltaE2000_pair_lab(result_lab, target_lab):
    """`target_lab` will be resized to `result_lab`'s (W, H) if different."""
    if target_lab.shape != result_lab.shape:
        # cv2.resize expects (W, H)
        target_lab = cv2.resize(
            target_lab, (result_lab.shape[1], result_lab.shape[0])
        )
    return float(deltaE_ciede2000(result_lab, target_lab).mean())


def deltaE2000_pair(image1_path, image2_path):
    """Return the mean CIEDE2000 between two image files (single float)."""
    result_lab = _bgr2lab(image1_path)
    target_lab = _bgr2lab(image2_path)
    return _deltaE2000_pair_lab(result_lab, target_lab)


def deltaE2000_folders(result_folder, content_folder, out_path=None):
    result_files = sorted(os.listdir(result_folder))
    content_files = sorted(os.listdir(content_folder))
    assert len(result_files) == len(content_files), (
        f'file count mismatch: {len(result_files)} vs {len(content_files)}'
    )

    scores = []
    _open_out(out_path)
    fout = open(out_path, 'w') if out_path else None
    try:
        pbar = tqdm(zip(result_files, content_files),
                    total=len(result_files), unit='file')
        for r_name, c_name in pbar:
            r_path = os.path.join(result_folder, r_name)
            c_path = os.path.join(content_folder, c_name)
            try:
                delta_e = deltaE2000_pair(r_path, c_path)
            except Exception as e:
                print(f'Error processing file {r_path}: {e}')
                continue
            scores.append(delta_e)
            if fout is not None:
                fout.write(f'{r_path}\t{delta_e:.4f}\n')
            pbar.set_description(
                'Avg DeltaE2000: {0:.4f}'.format(float(np.mean(scores)))
            )
        mean = float(np.mean(np.asarray(scores))) if scores else float('nan')
        if fout is not None:
            fout.write(f'Avg\t{mean:.4f}\n')
    finally:
        if fout is not None:
            fout.close()
    return {'per_image_mean': scores, 'overall_mean': mean}


def _read_list(list_path):
    rows = []
    list_dir = os.path.dirname(os.path.abspath(list_path))
    with open(list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            paths = [
                path if os.path.isabs(path)
                else os.path.normpath(os.path.join(list_dir, path))
                for path in parts[:4]
            ]
            rows.append(dict(zip(('content', 'reference', 'gt', 'pred'), paths)))
    return rows


def _open_out(out_path):
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)


def deltaE2000_list(list_path, out_path):
    rows = _read_list(list_path)
    print(f'read {len(rows)} rows from {list_path}')

    _open_out(out_path)
    scores = []
    with open(out_path, 'w') as f:
        f.write(f'# metric: deltaE2000 (cv2 BGR2Lab + skimage.deltaE_ciede2000, pred vs gt)\n')
        f.write(f'# list: {list_path}\n')
        f.write(f'# pairs: {len(rows)}\n')
        pbar = tqdm(rows, total=len(rows), unit='pair')
        for row in pbar:
            try:
                delta_e = deltaE2000_pair(row['pred'], row['gt'])
            except Exception as e:
                print(f"Error processing file {row['pred']}: {e}")
                continue
            scores.append(delta_e)
            f.write(f"{row['pred']}\t{delta_e:.4f}\n")
            pbar.set_description(
                'Avg DeltaE2000(pred, gt): {0:.4f}'.format(float(np.mean(scores)))
            )
        mean = float(np.mean(np.asarray(scores))) if scores else float('nan')
        f.write(f'Avg\t{mean:.4f}\n')

    print(f'wrote {out_path}')
    print(f'Avg CIEDE2000: {mean:.4f}')


def main():
    parser = argparse.ArgumentParser(description='CIEDE2000 color difference between images.')
    parser.add_argument('--image1', type=str, default=None, help='First image (single-pair mode).')
    parser.add_argument('--image2', type=str, default=None, help='Second image (single-pair mode).')
    parser.add_argument('--result-folder', type=str, default=None, help='Folder of result images.')
    parser.add_argument('--content-folder', type=str, default=None, help='Folder of reference images.')
    parser.add_argument('--list', type=str, default=None,
                        help='4-column txt list: content reference gt pred.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output txt with per-pair scores + average (list mode).')
    args = parser.parse_args()

    pair_mode = args.image1 is not None and args.image2 is not None
    folder_mode = args.result_folder is not None and args.content_folder is not None
    list_mode = args.list is not None

    if sum([pair_mode, folder_mode, list_mode]) != 1:
        parser.error('Provide exactly ONE of: --image1/--image2, --result-folder/--content-folder, --list.')

    if list_mode:
        if args.out is None:
            parser.error('--out is required with --list.')
        if not os.path.exists(args.list):
            raise FileNotFoundError(args.list)
        deltaE2000_list(args.list, args.out)
        return

    if pair_mode:
        for p in (args.image1, args.image2):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        delta_e = deltaE2000_pair(args.image1, args.image2)
        print('-' * 80)
        print('image1: {0}'.format(args.image1))
        print('image2: {0}'.format(args.image2))
        print('-' * 80)
        print('DeltaE2000  mean: {0:.4f}'.format(delta_e))
    else:
        for p in (args.result_folder, args.content_folder):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        res = deltaE2000_folders(args.result_folder, args.content_folder, args.out)
        print('-' * 80)
        print('Avg CIEDE2000: {0:.4f}'.format(res['overall_mean']))


if __name__ == '__main__':
    main()
