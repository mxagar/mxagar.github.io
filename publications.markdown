---
layout: page
title: Publications
permalink: /publications/
---

You can find an **updated** list of my publications on my [Google Scholar profile][google-scholar]. A **not necessarily updated** list is provided below:

<br>

{% for paper in site.data.publications %}
{{ paper.author }}. <b>{{ paper.title }}</b>. <i>{{ paper.journal }}</i>, {{ paper.year }}. {% if paper.doi != "" %} {{ paper.doi }}. {% endif %} {% if paper.note != "" %} {{ paper.note }}. {% endif %} {% if paper.link != "" %} <a href="{{ paper.link }}">LINK</a>. {% endif %}<br>
{% endfor %}

[google-scholar]: https://scholar.google.com/citations?user=DAP30jYAAAAJ
