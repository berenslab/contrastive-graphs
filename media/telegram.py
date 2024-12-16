import json
import subprocess
import sys
from pathlib import Path

from PIL.PngImagePlugin import PngImageFile, PngInfo

import telegram


class MscTelegram:
    def __init__(self, token=None, chat_id=None):
        with open(Path(__file__).parent / "telegram.json") as f:
            self.rc = json.load(f)

        self.token = token if token is not None else self.rc["token"]
        self.chat_id = chat_id if chat_id is not None else self.rc["chat_id"]
        self.bot = telegram.Bot(token=self.token)

    def send(self, fname):
        fname = Path(fname)
        ext = fname.suffix[1:]
        if ext == "mp4":
            return self.send_mp4(fname)
        elif ext == "png":
            return self.send_png(fname)
        elif ext == "txt" or ext == "md":
            return self.send_txt(fname)
        else:
            raise RuntimeError(
                "File ext '{}' not understood, file {} not sent".format(
                    ext, fname
                )
            )

    def send_mp4(self, fname):
        # how to get video info example taken from
        # https://github.com/kkroening/ffmpeg-python/blob/master/examples/video_info.py
        import ffmpeg

        probe = ffmpeg.probe(fname)

        video_stream = next(
            (
                stream
                for stream in probe["streams"]
                if stream["codec_type"] == "video"
            ),
            None,
        )

        if video_stream is None:
            print("No video stream found", file=sys.stderr)
            sys.exit(1)

        width = int(video_stream["width"])
        height = int(video_stream["height"])
        duration = float(video_stream["duration"])

        ffmpeg_proc = subprocess.run(
            [
                "ffmpeg",
                "-i",
                fname,
                "-c",
                "copy",
                "-map_metadata",
                "0",
                "-map_metadata:s:v",
                "0:s:v",
                "-f",
                "ffmetadata",
                "-",
            ],
            encoding="utf-8",
            stdout=subprocess.PIPE,
        )

        comment = f"`{fname}`"  # make sure that the variabel exists

        with open(fname, "rb") as f:
            return self.bot.send_animation(
                chat_id=self.chat_id,
                animation=f,
                width=width,
                height=height,
                duration=duration,
                parse_mode="Markdown",
                caption=comment,
            )

    def send_png(self, fname):
        png = PngImageFile(fname)

        png.load()  # load metadata
        with open(fname, "rb") as f:
            return self.bot.send_photo(
                chat_id=self.chat_id,
                photo=f,
                parse_mode="Markdown",
                caption=png.info.get("Comment", "`{}`".format(fname.name)),
            )

    def send_txt(self, fname):
        with open(fname) as f:
            text = f.read()

        return self.bot.send_message(
            chat_id=self.chat_id, text=text, parse_mode="Markdown"
        )


if __name__ == "__main__":
    bot = MscTelegram()
    bot.send(sys.argv[1])
