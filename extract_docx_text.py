import zipfile
import re
from pathlib import Path
files = [
    "Advanced_Brain_Tumor_Detection_Research_Seminar_Notes.docx",
    "Explainable_ViT_Brain_Tumor_Detection_Roadmap.docx",
]
for fn in files:
    path = Path(fn)
    print('FILE:', path.name)
    with zipfile.ZipFile(path, 'r') as z:
        text = ''
        for name in z.namelist():
            if name.startswith('word/document') or name.startswith('word/header') or name.startswith('word/footer'):
                data = z.read(name).decode('utf-8', errors='ignore')
                data = re.sub(r'<[^>]+>', ' ', data)
                text += data + '\n'
        print(text[:15000])
        print('--- END ---\n')
