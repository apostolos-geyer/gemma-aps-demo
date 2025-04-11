# pyright: basic

import enum
import os
import re
import sys
import typing

import nltk
import rich
import rich.markdown
import structlog
import torch
import typer
from dotenv import load_dotenv
from nltk.downloader import Downloader
from transformers.pipelines import pipeline

downloader = Downloader()
pkg = "punkt_tab"

if not (downloader.status(pkg, nltk.data.path[0]) == downloader.INSTALLED):
    downloader.download(pkg)


app = typer.Typer()

log: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger()

load_dotenv()

generator = pipeline(
    "text-generation",
    "google/gemma-7b-aps-it",
    device="mps",
    torch_dtype=torch.bfloat16,
    token=os.environ["HUGGINGFACE_HUB_TOKEN"],
)


console = rich.console.Console()


@app.command("main")
def main():
    """
    simple aps pipeline
    """

    user_input = "".join([line for line in sys.stdin])
    messages = [{"role": "user", "content": process_proposition_input(user_input)}]

    console.clear()

    console.log(f"[green]message:[/green] {user_input}")
    with console.status("running APS"):
        generated_output = typing.cast(
            list[dict[str, str]],
            generator(messages, max_new_tokens=4096, return_full_text=False),
        )
        console.log("processing output")
        result: list[list[str]] = process_proposition_output(
            generated_output[0]["generated_text"]
        )

    console.log("[green]extracted propositions[/green]")
    output_formatted = ""
    n = len(result)
    for i, grp in enumerate(result):
        output_formatted += f"## Group {i}\n"
        for prop in grp:
            output_formatted += f" - {prop}\n"
        if i != (n - 1):
            output_formatted += "\n---\n"
    console.print(rich.markdown.Markdown(output_formatted))


class Marker(enum.StrEnum):
    START = "<s>"
    END = "</s>"
    SEPARATOR = "\n"


def process_proposition_input(text: str) -> str:
    """preprocess text as input for the proposition segmentation"""
    input_sents = nltk.tokenize.sent_tokenize(text)
    propositions_input = Marker.SEPARATOR.join(
        [f"{Marker.START} {sent} {Marker.END}" for sent in input_sents]
    ).strip(Marker.SEPARATOR)

    console.log(
        "create_proposition_input",
        dict(
            text=text,
            input_sents=input_sents,
            propositions_input=propositions_input,
        ),
    )

    return propositions_input


OUTPUT_PATTERN: re.Pattern = re.compile(
    f"{re.escape(Marker.START)}(.*?){re.escape(Marker.END)}", re.DOTALL
)


def process_proposition_output(text: str):
    output_groups = re.findall(OUTPUT_PATTERN, text)
    predicted_proposition_groups = []
    for group in output_groups:
        group = group.strip(Marker.SEPARATOR)
        propositions = [
            groupmember[2:] for groupmember in group.split(Marker.SEPARATOR)
        ]
        predicted_proposition_groups.append(propositions)

    console.log(
        "process_proposition_output",
        dict(
            text=text,
            output_groups=output_groups,
            predicted_proposition_groups=predicted_proposition_groups,
        ),
    )
    return predicted_proposition_groups


if __name__ == "__main__":
    app()
