"""Download a ready-made text corpus into corpus.txt.

Run:   python download_corpus.py                 (Shakespeare, about 1 MB)
       python download_corpus.py --name sonnets  (Shakespeare's sonnets, a smaller poem corpus)

You can also skip this and put ANY plain-text file (poems, song lyrics, code) at corpus.txt,
or pass your own file:  python train.py --corpus my_lyrics.txt
"""
import argparse
import sys
import urllib.request

from common import CORPUS_FILE

SOURCES = {
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "sonnets": "https://www.gutenberg.org/cache/epub/1041/pg1041.txt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=sorted(SOURCES), default="shakespeare")
    args = parser.parse_args()

    url = SOURCES[args.name]
    print(f"Downloading {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8", errors="ignore")
    except Exception as error:
        sys.exit(f"Download failed: {error}\n"
                 "Open the link in your browser, save the page as corpus.txt in this folder, and go on to train.py.")

    if args.name == "sonnets":                     # cut the Project Gutenberg header and footer
        start = text.find("THE SONNETS")
        end = text.find("End of the Project Gutenberg")
        if start != -1 and end != -1:
            text = text[start:end]

    CORPUS_FILE.write_text(text, encoding="utf-8")
    print(f"Saved {len(text):,} characters to {CORPUS_FILE}")
    print("Next step:  python train.py")


if __name__ == "__main__":
    main()
