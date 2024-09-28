from PIL import Image
from pathlib import Path

def single(source: Path, size: int = 600):
    dest_path: Path = source.parent / (source.stem + "_thumbnail" + source.suffix)

    image = Image.open(source)
    image.thumbnail((size,size), Image.Resampling.LANCZOS)

    image.save(dest_path)

    print(f"resized {source}")

def do_dir(source_dir: Path):
    print(source_dir)
    for s in source_dir.rglob("*"):
        if "thumbnail" in str(s):
            continue
        # print(s)
        single(s, 400)

def main():
    do_single: bool = True
    if do_single:
        single_source_path: Path = Path(f"../web/img/ski/MtHood/satellite/2024-03-07-00-00_2024-03-07-23 59_Sentinel-2_L2A_True_color.png")
        single(single_source_path, 600)
    else:
        source_dir: Path = Path(f"../web/img/ski/MtHood/2024-02-17")
        do_dir(source_dir)

if __name__ == "__main__":
    main()
