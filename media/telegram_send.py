import asyncio
import json
import subprocess
import sys
from pathlib import Path

import telegram
from PIL.PngImagePlugin import PngImageFile, PngInfo


async def send_mp4(fname, bot, chat_id):
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
        return bot.send_animation(
            chat_id=chat_id,
            animation=f,
            width=width,
            height=height,
            duration=duration,
            parse_mode="Markdown",
            caption=comment,
        )


async def send_png(fname, bot, chat_id):
    png = PngImageFile(fname)

    png.load()  # load metadata
    with open(fname, "rb") as f:
        return await bot.send_photo(
            chat_id=chat_id,
            photo=f,
            parse_mode="Markdown",
            caption=png.info.get("Comment", "`{}`".format(fname.name)),
        )


async def send_txt(fname, bot, chat_id):
    with open(fname) as f:
        text = f.read()

    return bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")


async def send(fname, bot, chat_id):
    fname = Path(fname)
    ext = fname.suffix[1:]
    if ext == "mp4":
        return await send_mp4(fname, bot, chat_id)
    elif ext == "png":
        return await send_png(fname, bot, chat_id)
    elif ext == "txt" or ext == "md":
        return await send_txt(fname, bot, chat_id)
    else:
        raise RuntimeError(
            "File ext '{}' not understood, file {} not sent".format(ext, fname)
        )


async def main():
    with open(Path(__file__).parent / "telegram.json") as f:
        rc = json.load(f)

    bot = telegram.Bot(token=rc["token"])
    async with bot:
        await send(sys.argv[1], bot, rc["chat_id"])


if __name__ == "__main__":
    asyncio.run(main())
