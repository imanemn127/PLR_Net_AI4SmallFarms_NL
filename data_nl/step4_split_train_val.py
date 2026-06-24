import os, random

PATCHES_DIR = "/mnt/DATA/IMANE/PLR-Net/data_nl/patches/train/images"
OUT_DIR = "/mnt/DATA/IMANE/PLR-Net/data_nl/patches"

# Lister tous les noms de fichiers sans extension .tif
names = sorted(os.path.splitext(f)[0] for f in os.listdir(PATCHES_DIR) if f.endswith(".tif"))
random.seed(42)
random.shuffle(names)
split = int(0.8 * len(names))

with open(os.path.join(OUT_DIR, "train.txt"), "w") as f:
    f.write("\n".join(names[:split]))
with open(os.path.join(OUT_DIR, "val.txt"), "w") as f:
    f.write("\n".join(names[split:]))

print(f"Train: {split} patches, Val: {len(names)-split} patches")