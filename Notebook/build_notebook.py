import json, re

src = open("/home/claude/HousePricePrediction/Notebook/pipeline.py").read()
blocks = re.split(r"^# %%.*$", src, flags=re.MULTILINE)[1:]
headers = re.findall(r"^# %%.*$", src, flags=re.MULTILINE)

cells = []
for header, block in zip(headers, blocks):
    block = block.strip("\n")
    if "[markdown]" in header:
        lines = [l[2:] if l.startswith("# ") else l[1:] if l.startswith("#") else l
                 for l in block.split("\n")]
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [l + "\n" for l in lines][:-1] + [lines[-1]] if lines else [],
        })
    else:
        lines = block.split("\n")
        while lines and lines[0] == "":
            lines.pop(0)
        while lines and lines[-1] == "":
            lines.pop()
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [l + "\n" for l in lines][:-1] + [lines[-1]] if lines else [],
        })

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("/home/claude/HousePricePrediction/Notebook/house_price_prediction.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Wrote notebook with", len(cells), "cells")
