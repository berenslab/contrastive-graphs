import sys
from pathlib import Path

from matplotlib import font_manager


def main():
    fonts = (
        p / f
        for p in Path("ttf").glob("*")
        for f in (p / "all-fonts").read_text().split("\n")
        if f != ""
    )
    [font_manager.fontManager.addfont(f) for f in fonts]
    font_manager.json_dump(font_manager.fontManager, sys.argv[1])


if __name__ == "__main__":
    main()
