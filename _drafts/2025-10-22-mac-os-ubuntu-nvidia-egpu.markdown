---
layout: post
title:  "My Personal eGPU Server Setup"
subtitle: "How to Run and Train LLMs Locally with NVIDIA Chips from a Mac & Linux Setup"
date:   2025-10-21 12:30:00 +0200
categories: eGPU, AI engineering, LLM, machine learning, Ollama, Remote VS Code, Linux, deep learning, model training, model inference
permalink: /blog/mac-os-ubuntu-nvidia-egpu.html
comments: true
---
<h1 style="color:grey;font-style:italic">{% if page.subtitle %}{{ page.subtitle }}{% endif %}
</h1>

<div style="line-height:150%;">
    <br>
</div>

<p align="center">
<img src="/assets/linux_nvidia_egpu/workstation-dgx-spark-nvidia.jpg" alt="NVIDIA DGX Spark" width="1000"/>
<small style="color:grey">This blog post is not about the <a href="https://www.nvidia.com/en-us/products/workstations/dgx-spark/">NVIDIA DGX Spark</a>. Instead, it's about my eGPU setup, the <i>personal supercomputer</i> I've been using the past 2 years. Image from <a href="https://nvidianews.nvidia.com/news/nvidia-dgx-spark-arrives-for-worlds-ai-developers">NVIDIA</a>.
</small>
</p>

<div style="height: 20px;"></div>
<div align="center" style="border: 1px solid #e4f312ff; background-color: #fcd361b9; padding: 1em; border-radius: 6px;">
<strong>
For a detailed setup guide, check <a href="https://github.com/mxagar/linux_nvidia_egpu">this GitHub repository</a>.
</strong>
</div>
<div style="height: 30px;"></div>

You have maybe followed the release of the [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) *personal supercomputer*. The device, with 128 GB of memory, 20 CPU cores, and a price of USD $3,999.00, will be definitely on the wish list of any AI nerd for this Christmas.

This blog post is my personal and humble alternative. Indeed, in the past 2 years I have been using an NVIDIA eGPU (external GPU) from my MacBook Pro M1, but via a Linux machine, which plays the role of a server. Since some colleagues and friends showed interest, I decided to [thoroughly document it on GitHub](https://github.com/mxagar/linux_nvidia_egpu) in the form of the guide I was looking for, but couldn't find completely. On the other hand, this blog post introduces the overall setup and the motivation behind it. Here's the schematics of my *supercomputer*:

<p align="center">
<img src="/assets/linux_nvidia_egpu/egpu_linux.png" alt="eGPU Linux & Mac Setup" width="1000"/>
<small style="color:grey">My eGPU setup consists of a MacBook M1 and a Linux server with an NVIDIA eGPU.
</small>
</p>

I mainly use the eGPU to train general Deep Learning models (with [VS Code Remote Development](https://code.visualstudio.com/docs/remote/ssh)) and to run LLMs locally (with [Ollama](https://ollama.com/)); as illustrated in the figure above:

- I have a [Lenovo ThinkPad P14s](https://www.lenovo.com/gb/en/p/laptops/thinkpad/thinkpadp/p14s-amd-g1/22wsp144sa1) with an integrated NVIDIA Quadro T500 graphics card running Ubuntu.
- I attach to a Thunderbolt port of the Lenovo a [Razer Core X External Case](https://www.razer.com/mena-en/gaming-laptops/razer-core-x) which contains a [NVIDIA GeForce RTX 3060](https://www.gigabyte.com/Graphics-Card/GV-N3060GAMING-OC-12GD-rev-20), with 12 GB of memory.
- I run applications which require GPU power on the Lenovo/Ubuntu but interface with them via my MacBook Pro M1.

You might ask *why I would want to run and train models locally*, since we have many cloud services available that spare us the hassle. Here're my answers:

- Many models (LLMs or any other DL networks) can be used locally for a **fraction of the cost** required by cloud providers; in fact, the [NVIDIA RTX 3060](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/) with 12 GB is quite similar to the often offered low tier GPU, the [NVIDIA T4](https://www.nvidia.com/en-us/data-center/tesla-t4/).
- Use data **confidentially**: running models locally allows you to process sensitive or proprietary data (e.g., personal notes, internal reports, or corporate documents) without uploading them to third-party servers. This means full control over your data lifecycle, compliance with privacy policies, and peace of mind knowing that no external provider logs or stores your content.
- **Avoid dependence** on cloud services: while cloud platforms provide flexibility, they also create a point of failure and an ongoing dependency on external infrastructure and pricing. Outages like the [AWS downtime of December 2021](https://techcrunch.com/2021/12/07/amazon-web-services-went-down-and-took-a-bunch-of-the-internet-with-it/) or the more recent [AWS outage in October 2025](https://www.wired.com/story/what-that-huge-aws-outage-reveals-about-the-internet/) show how fragile these systems can be.
- **Learn** how to set up hardware, firmware, and software: managing your own GPU infrastructure provides a deeper understanding of the systems that power modern AI. From BIOS configuration and driver setup to Docker and Conda environments, each layer teaches valuable skills that translate directly into real-world MLOps and engineering practice.

You might also ask *why not stick to a single computer, Ubuntu or MacOS, with an attached eGPU*. That question has several layers:

- Even though I really like Ubuntu, MacOS offers in my opinion another level of user experience in general, which I find more polished than the Linux variant.
- In the past, Intel-based Macs supported AMD eGPUs, but since the introduction of the Apple M1, that option seems to have vanished. 
- Ideally, I'd use MacOS with NVIDIA eGPU support, because NVIDIA chips are the industry standard.
- Another option would be to upgrade my MacBook Pro M1 to a MacStudio M3 Ultra or similar, which comes with a very powerful processor &mdash; but why abandon a perfectly capable MacBook Pro M1?

## Setup Guide: A Summary

The [GitHub repository I have created](https://github.com/mxagar/linux_nvidia_egpu) covers all the questions and technical steps necessary to get up and running an eGPU:

- [Hardware requirements](https://github.com/mxagar/linux_nvidia_egpu/tree/main?tab=readme-ov-file#step-0-hardware-requirements): *What hardware do I need for an eGPU setup? Which GPUs and enclosures are compatible? How much VRAM do typical ML models require?*
- [Installation of Ubuntu](https://github.com/mxagar/linux_nvidia_egpu/tree/main?tab=readme-ov-file#step-1-install-ubuntu) and [NVIDIA libraries](https://github.com/mxagar/linux_nvidia_egpu/tree/main?tab=readme-ov-file#step-3-install-and-configure-nvidia-and-gpu-related-libraries): *How do I install and configure Ubuntu so it works smoothly with my external NVIDIA GPU?*

Additionally, some extra but very practical aspects are covered in dedicated sections:

- [Installation of Docker with GPU support](https://github.com/mxagar/linux_nvidia_egpu/tree/main?tab=readme-ov-file#step-5-install-docker-with-nvidia-gpu-support): Containerization has become crucial in the AI/ML industry; unfortunately, setting up deep learning or model-serving workloads inside images with full GPU acceleration is sometimes not that straightforward &mdash; fear not: a simple and reliable recipe is provided in this section.
- [Remote access configuration](https://github.com/mxagar/linux_nvidia_egpu/tree/main?tab=readme-ov-file#step-6-remote-access-configuration): This section explains how to securely connect to the Ubuntu GPU machine from another device (e.g., a MacBook) within the same local network.

After the installation, we should be able to check our NVIDIA eGPU via the Terminal on the Mac.

<p align="center">
<img src="/assets/linux_nvidia_egpu/mac_nvidia_smi.png" alt="MacOS NVIDIA SMI" width="1000"/>
<small style="color:grey">Snapshot of the <code>nvidia-smi</code> output on the Ubuntu machine (hostname: <code>urgull</code>) but executed from my MacBook (hostname: <code>kasiopeia</code>). We can see the eGPU and its load: NVIDIA GeForce RTX 3060, 14W / 170W used, 26MiB / 12288MiB used.
</small>
</p>

## Using the eGPU: Remote VS Code and Ollama

Once we get the correct output from `nvidia-smi`, we can start using the eGPU. To that end, the [guide GitHub repository](https://github.com/mxagar/linux_nvidia_egpu) contains a simple Jupyter notebook we can run: [test_gpu.ipynb](https://github.com/mxagar/linux_nvidia_egpu/blob/main/test_gpu.ipynb).

The way I prefer to run complete repositories remotely (i.e., on the Ubuntu machine but interfaced from the MacBook) is using a [**Remote VS Code**](https://code.visualstudio.com/docs/remote/ssh) instance. To start one, these are the preliminary steps we need to follow:

1. Open the MacBook Terminal (make sure no VPN connections are active).
2. SSH to the Ubuntu machine with our credentials.
3. Clone the [GitHub repository](https://github.com/mxagar/linux_nvidia_egpu) with the notebook [test_gpu.ipynb](https://github.com/mxagar/linux_nvidia_egpu/blob/main/test_gpu.ipynb).
4. Install the GPU Conda environment.

Steps 1-4 are carried out with these commands:

```bash
# -- MacBook
ssh <username>@<hostname-ubuntu>.local
ssh mikel@urgull.local

# -- Ubuntu via MacBook
cd && mkdir -p git_repositories && cd git_repositories
git clone https://github.com/mxagar/linux_nvidia_egpu.git
conda env create -f conda.yaml
```

Then, we can start a remote VS Code instance:

5. We open VS Code on our MacBook.
6. Click on *Open Remote Window* (bottom left corner) > *Connect to Host...*.
7. We enter the user and host as in `<username>@<hostname-ubuntu>.local`, followed by the password.

... *et voilà*: we have already a VS Code instance running on the Ubuntu machine, but interfaced by the MacBook UI! Now, we can open any folder, including the folder containing the notebook:

8. Click on *Explorer menu* (left menu bar) > *Open Folder*.
9. And, finally, load our repository cloned in `~/git_repositories/linux_nvidia_egpu`.

Once we select the `gpu` environment kernel for the [test_gpu.ipynb](https://github.com/mxagar/linux_nvidia_egpu/blob/main/test_gpu.ipynb) notebook, we can start executing its cells. Among others, a simple CNN is trained with the MNIST dataset (~45MB) and *the NVIDIA RTX 3060 is almost 2x faster than the MacBook Pro M1* (37 sec. vs. 62 sec.). In terms of memory, my MacBook has a *unified memory* of 16GB, vs. the 12GB VRAM of the RTX 3060; it might seem that the NVIDIA chip is worse, but in practice it is not: I can load larger models using its dedicated memory, because the Apple memory is shared between CPU and GPU, which can lead to bottlenecks.

<p align="center">
<img src="/assets/linux_nvidia_egpu/mac_ubuntu_egpu_vscode.png" alt="VS Code Remote Window" width="1000"/>
<small style="color:grey">Snapshot of the remote VS Code instance: the repository is on the Ubuntu machine with leveraging the eGPU (hostname: <code>urgull</code>), but interfaced from my MacBook (hostname: <code>kasiopeia</code>).
</small>
</p>

<div style="height: 20px;"></div>
<p align="center">── ◆ ──</p>
<div style="height: 20px;"></div>

Another application I use quite extensively with the eGPU is [**Ollama**](https://ollama.com/), which enables *local* Large Language Models (LLMs) for a plethora of tasks.

To run Ollama on the eGPU but interfaced from the MacBook, first, we need to [install it](https://github.com/mxagar/linux_nvidia_egpu?tab=readme-ov-file#ollama-server-use-ollama-llms-running-on-the-gpu-ubuntu-but-from-another-machine) properly on both machines. Then, we can follow these easy commands to start a chatbot via the CLI:

```bash
# -- Ubuntu (...or via ssh from MacBook)
# Make sure Ollama uses GPU and is accessible in our LAN
export OLLAMA_USE_GPU=1
export OLLAMA_HOST=0.0.0.0:11434
# Download a 9.1GB model (takes some minutes) and start the Ollama server
ollama pull gemma3:12b
ollama serve &

# -- MacBook
# Change the Ollama host to the Ubuntu machine
export OLLAMA_HOST=urgull.local:11434
ollama run gemma3:12b
# ... now we can chat :)

# To revert to use the local MacBook Ollama service
export OLLAMA_HOST=127.0.0.1:11434
```

The result is summarized on the following snapshot:

<p align="center">
<img src="/assets/linux_nvidia_egpu/mac_ubuntu_egpu_ollama.png" alt="Ollama on eGPU" width="1000"/>
<small style="color:grey"><a href="https://deepmind.google/models/gemma/gemma-3/">Gemma 3 12B</a> running on the Ubuntu eGPU via Ollama, but operated from the MacBook.
</small>
</p>

The Ollama server can be also reached from our LAN using `cURL`:

```bash
curl http://urgull.local:11434/api/generate -d '{
  "model": "gemma3:12b",
  "prompt": "Write a haiku about machine learning."
}'
```

Of course, there are many useful and more interesting downstream applications:

- Private chatbot with proper GUI, history, document upload and internet access
- Copilot-style code completion
- Agents: Autonomous CLI Agents, Local Operator, Ollama MCP Agent, etc.
- ...

However, those applications are not in the scope of this post &mdash; maybe I will introduce them in another one ;)

## Wrap Up

In this post, I've shared the motivation and architecture behind my personal eGPU server setup &mdash; a compact yet powerful alternative to commercial "AI workstations". The combination of a Linux GPU server (NVIDIA RTX 3060) and a MacBook (Pro M1) client creates a seamless environment for experimentation: you can train models efficiently, run large LLMs locally via Ollama, and enjoy the responsive UI and ecosystem of macOS for daily work.

Running models locally is not only cost-effective but also empowers you to work autonomously, privately, and creatively, without depending on cloud services.

<br>

> Would you prefer to build your own local AI workstation, or do you trust cloud services enough to rely on them entirely? What would be your ideal balance between local and cloud compute?

<br>

If you're interested in a step-by-step guide, check [**my Github repository of the project**](https://github.com/mxagar/linux_nvidia_egpu).
