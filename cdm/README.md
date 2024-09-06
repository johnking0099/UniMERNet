<div align="center">

English | [简体中文](./README-CN.md)

<h1>CDM: A Reliable Metric for Fair and Accurate Formula Recognition</h1>

[[ Paper ]](https://arxiv.org/pdf/2409.03643) [[ Website ]](https://github.com/opendatalab/UniMERNet/tree/main/cdm)
[[Demo 🤗(Hugging Face)]](https://huggingface.co/spaces/opendatalab/CDM-Demo)

</div>


# Overview

CDM's architecture and comparison with other metrics:

<div align="center">
    <img src="assets/demo/demo.png" alt="Overview" width="42.2%" style="margin-right: 2px">
    <img src="assets/demo/cases.png" alt="Demo" width="52%">
</div>

# 

# Usage

## Try Online Demo

Try CDM on our online demo: [(Hugging Face)🤗](https://huggingface.co/spaces/opendatalab/CDM-Demo)

## Install CMD Locally

Given CDM's complex environment dependencies, we recommend trying it on Linux systems.

## prepare environment

Nodejs, imagemagic, pdflatex are requried packages when render pdf files and convert them to images, here are installation guides.

### step.1 install nodejs

```
wget https://registry.npmmirror.com/-/binary/node/latest-v16.x/node-v16.13.1-linux-x64.tar.gz

tar -xvf node-v16.13.1-linux-x64.tar.gz

mv node-v16.13.1-linux-x64/* /usr/local/nodejs/

ln -s /usr/local/nodejs/bin/node /usr/local/bin

ln -s /usr/local/nodejs/bin/npm /usr/local/bin

node -v
```

### step.2 install imagemagic

the version of imagemagic installed by `apt-gt` usually be 6.x, so we also install it from source code.
```
git clone https://github.com/ImageMagick/ImageMagick.git ImageMagick-7.1.1

cd ImageMagick-7.1.1

./configure

make

sudo make install

sudo ldconfig /usr/local/lib

convert --version
```

### step.3 install latexpdf

```
apt-get update

sudo apt-get install texlive-full
```

### step.4 install python requriements

```
pip install -r requirements.txt
```


## Use CDM Locally

Should the installation goes well, you may now use CDM to evaluate your formula recognition results.

### 1. batch evaluation 

```
python evaluation.py -i {path_to_your_input_json}
```

the format of input json like this:
```
[
    {
        "img_id": "case_1",      # optional key
        "gt": "y = 2z + 3x",
        "pred": "y = 2x + 3z"
    },
    {
        "img_id": "case_2",
        "gt": "y = x^2 + 1",
        "pred": "y = x^2 + 1"
    },
    ...
]
```

### 2. launch a gradio demo

```
python app.py
```