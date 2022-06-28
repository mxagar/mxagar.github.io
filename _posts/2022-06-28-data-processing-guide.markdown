---
layout: post
title:  "Practical Recipes for Your Data Processing"
subtitle: "The 80/20 Guide that Solves Your Data Cleaning, Exploratory Data Analysis and Feature Engineering with Tabular Datasets"
date:   2022-06-28 09:30:00 +0200
categories: data science, data analysis, exploratory data analysis, feature engineering, data modelling, hypothesis testing, regression, classification, random forests, summary
permalink: /blog/data-processing-guide.html
comments: true
---

<h1 style="color:grey;font-style:italic">{% if page.subtitle %}{{ page.subtitle }}{% endif %}
</h1>

<p align="center">
<img src="/assets/data_processing_guide/tim-gouw-1K9T5YiZ2WU-unsplash.jpg" alt="Donostia-San Sebastian: Photo by @ultrashricco from Unsplash" width="1000"/>
<small style="color:grey">Don't worry, working hard often pays off. Photo by <a href="https://unsplash.com/photos/1K9T5YiZ2WU">Tim Gouw from Unsplash</a>.</small>
</p>

Thanks to the powerful packages we have available nowadays, training machine learning models is often a very tiny step in the pipeline of a regular data science project. Altogether, we need to address the following tasks:

1. Data Understanding & Formulation of the Questions
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Feature Selection
6. Data Modelling

Additionally, if life inferences are planned, several parts from the steps 2-5 needs to be prepared for production environments, i.e., they need to be transferred into scripts in which reproducibility and maintainability can be guaranteed for robust and trustworthy deployments.

Independently from that fact and remaining still in the research and development environment, steps 2-5 consume a large percentage of the effort. We need to apply some kind of methodical creativity to often messy datasets that almost never behave as we initially want.

So, is there an easy way out? Unfortunately, I'd say there is not, at least I don't know one yet. However, **I have collected a series of guidelines and code snippets you can use systematically to ease your data processing journey in a [Github repository](https://github.com/mxagar/eda_fe_summary)**. It summarizes the map I have sketched along the years.

In the repository, you will find two important files:

- A large python script `data_processing.py` which contains many code examples; these cover 80% of the processing techniques I usually apply on *tabular* datasets.
- The `README.md` itself, which sums up the steps and *dos & don'ts* in the standard order for data processing described above.

Some caveats:

- The script `data_processing.py` does not run! Instead, it's a compilation of useful commands with comments.
- I assume the reader knows the topic, i.e., the repository is not for complete beginners.
- The guide does not cover advanced cases either: it's a set of tools that follow the 80/20 [Pareto principle](https://en.wikipedia.org/wiki/Pareto_principle).
- The guide focuses on *tabular* data; images and text have their own particular pipelines, not covered here.
- This is my personal guide, made for me; no guarantees are assured and it will probably change organically.

<br>

> Do you find the repository helpful? What would you add? Do you know similar summaries to learn from?

<br>

{% if page.comments %} 
{% include disqus-comments.html %}
{% endif %}
