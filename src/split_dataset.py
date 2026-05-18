import argparse
import random
import shutil
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    random.seed(seed)

    if not source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {source_dir}')

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('Train, validation, and test ratios must sum to 1.0.')

    for subset in ['train', 'val', 'test']:
        subset_dir = output_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(source_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        images = [p for p in class_dir.iterdir() if p.is_file()]
        random.shuffle(images)
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:],
        }

        for split_name, files in splits.items():
            target_dir = output_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for file_path in files:
                shutil.copy2(file_path, target_dir / file_path.name)

    print(f'Successfully split dataset from {source_dir} into {output_dir}')
    print('Train/val/test ratios:', train_ratio, val_ratio, test_ratio)


def parse_args():
    parser = argparse.ArgumentParser(description='Split raw image dataset into train/val/test folders')
    parser.add_argument('--source', required=True, help='Source folder containing class subfolders')
    parser.add_argument('--output', default='dataset', help='Output folder to create train/val/test splits')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    split_dataset(
        args.source,
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
