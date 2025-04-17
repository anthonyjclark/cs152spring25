#!/usr/bin/env python

"""
Run this file with:
GRADIO_SERVER_PORT=<port> python FrankOrFraryGradioApp.py </path/to/model>

Where
- <port> is your server port number
- </path/to/model> is the path to the model you want to use for inference
"""

import gradio as gr
from fastai.vision.all import *

# Load the trained model
path = Path(sys.argv[1])
model = load_learner(path)


def classify(img):
    label, label_index, probabilities = model.predict(img)
    print(label, label_index, probabilities)

    return {
        "Pomona": probabilities[0].item(),
        "Scripps": probabilities[1].item(),
        "CMC": probabilities[2].item()
    }


title = "Pomona, Scripps, or CMC? I'll Decide!"
website = "A demo for [CS 152](https://cs.pomona.edu/classes/cs152/)"

iface = gr.Interface(
    fn=classify,
    inputs="image",
    outputs="label",
    title=title,
    article=website,
).launch()
