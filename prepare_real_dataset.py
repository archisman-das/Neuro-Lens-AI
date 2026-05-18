import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
TUMOR_CLASSES = {'glioma', 'meningioma', 'pituitary'}
NO_TUMOR_CLASSES = {'notumor', 'no_tumor', 'no tumor', 'no-tumor'}


def image_files(directory):
    return sorted(path for path in directory.rglob('*') if path.suffix.lower() in IMAGE_EXTENSIONS)


def split_items(items, train_ratio, val_ratio):
    train_end = int(len(items) * train_ratio)
    val_end = train_end + int(len(items) * val_ratio)
    return {
        'train': items[:train_end],
        'val': items[train_end:val_end],
        'test': items[val_end:],
    }


def copy_split(files_by_split, output_dir, class_name):
    for split, files in files_by_split.items():
        target_dir = output_dir / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(files):
            target = target_dir / f'{class_name}_{index:05d}{source.suffix.lower()}'
            shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description='Prepare real MRI dataset as binary tumor/no_tumor splits.')
    parser.add_argument('--source', default='data_sources/Brain Tumor MRI')
    parser.add_argument('--output', default='dataset_real')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--max_per_class', type=int, default=0, help='Optional cap per binary class. 0 keeps all balanced data.')
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f'Source dataset folder not found: {source}')

    tumor_files = []
    no_tumor_files = []
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        normalized = class_dir.name.strip().lower().replace('-', '_').replace(' ', '_')
        files = image_files(class_dir)
        if normalized in TUMOR_CLASSES:
            tumor_files.extend(files)
        elif normalized in NO_TUMOR_CLASSES:
            no_tumor_files.extend(files)

    if not tumor_files or not no_tumor_files:
        raise ValueError(
            f'Expected tumor and no_tumor images. Found tumor={len(tumor_files)}, no_tumor={len(no_tumor_files)}.'
        )

    rng = random.Random(args.seed)
    rng.shuffle(tumor_files)
    rng.shuffle(no_tumor_files)

    class_limit = min(len(tumor_files), len(no_tumor_files))
    if args.max_per_class > 0:
        class_limit = min(class_limit, args.max_per_class)

    tumor_files = tumor_files[:class_limit]
    no_tumor_files = no_tumor_files[:class_limit]

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    copy_split(split_items(tumor_files, args.train_ratio, args.val_ratio), output, 'tumor')
    copy_split(split_items(no_tumor_files, args.train_ratio, args.val_ratio), output, 'no_tumor')

    print(f'Prepared {output}')
    print(f'Tumor images: {len(tumor_files)}')
    print(f'No tumor images: {len(no_tumor_files)}')
    for split in ['train', 'val', 'test']:
        tumor_count = len(list((output / split / 'tumor').glob('*')))
        no_tumor_count = len(list((output / split / 'no_tumor').glob('*')))
        print(f'{split}: tumor={tumor_count}, no_tumor={no_tumor_count}')


if __name__ == '__main__':
    main()
