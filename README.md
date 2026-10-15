# Mikel Sagardia's Personal Website

[mikelsagardia.io](https://mikelsagardia.io) · [mxagar.github.io](https://mxagar.github.io)

## Local preview

Use **Ruby 3.3.12** (`.ruby-version`) and **Bundler 4.0.23**. The `github-pages`
gem selects Jekyll 3.10.0 and the same theme/plugin versions as GitHub's native
Pages builder. Use `bundle exec` for all Jekyll commands.

```bash
cd .../mxagar.github.io
bundle exec jekyll serve
```

Open <http://localhost:4000>. Stop with **Ctrl+C**.
Changes normally rebuild automatically; restart after editing `_config.yml`.

## Write posts and drafts

Create a draft such as `_drafts/my-post.md`. Start with YAML front matter:

```yaml
---
layout: post
title: "My post title"
categories: [machine-learning, python]
permalink: /blog/my-post.html
comments: true
---
```

Write Markdown below the front matter. Store images under `assets/my-post/`
and reference them as `![Description](/assets/my-post/image.png)`.

Preview drafts with:

```bash
bundle exec jekyll serve --drafts
```

To publish, move the draft to `_posts/YYYY-MM-DD-my-post.md` and add its date
to the front matter, for example `date: 2026-10-15 12:00:00 +0200`.
Check the title, date, images, and permalink before committing. Keep the
permalink stable after publication so existing links continue to work.

Drafts are omitted from normal builds and deployment, although their source
files remain visible in this public repository. Keep unfinished posts in
`_drafts`: GitHub Pages enables future-dated posts, so a future date alone
does not prevent publication.

## Files and folders

| Path | Purpose |
| --- | --- |
| `_posts/` | Published Markdown posts, named `YYYY-MM-DD-slug.md` or `.markdown`. |
| `_drafts/` | Work in progress; preview with `--drafts`. |
| `pages/`, `index.md`, `publications.md` | Regular pages and the home page. |
| `assets/` | Images, downloadable files, and custom CSS/SCSS. |
| `_data/` | Structured data, including `publications.yaml`. |
| `_layouts/` | Page templates that override the theme's layouts. |
| `_includes/` | Reusable HTML fragments that override theme includes. |
| `_config.yml` | Site URL, theme, plugins, Markdown, and build settings. |
| `Gemfile`, `Gemfile.lock`, `.ruby-version` | Dependencies, exact resolved versions/platforms, and project Ruby. |
| `_site/` | Generated website; ignored, disposable, and recreated by Jekyll. Never edit it. |
| `vendor/bundle/`, `.bundle/` | Installed gems and local Bundler settings; ignored. |
| `.sass-cache/`, `.jekyll-cache/`, `.jekyll-metadata` | Generated caches; ignored and disposable. |
| `.git/` | Git history and repository metadata; keep it. |
| `.vscode/` | Optional local VS Code settings; ignored. |
| `CNAME` | GitHub Pages custom domain: `mikelsagardia.io`. |

## Theme: Minima 2.5.1

Minima is installed as a gem. With this setup, its files live at
`vendor/bundle/ruby/3.3.0/gems/minima-2.5.1` relative to the repository.

Find the active path with `bundle info minima --path`.

Make customizations in this repository's `_layouts/`, `_includes/`, or
`assets/css/custom.scss`; edits inside `vendor/` are lost on reinstall.

## Updates

If pulling changes updates `Gemfile` or `Gemfile.lock`, sync the installed gems
with the committed versions:

```bash
bundle install
```

To deliberately refresh dependencies within the Gemfile's constraints, run
these same commands from the repository root on either macOS or Linux:

```bash
bundle update
JEKYLL_ENV=production bundle exec jekyll build
bundle exec jekyll serve --drafts
```

`bundle update` writes the updated versions to `Gemfile.lock`. The shared
lockfile already includes macOS and Linux platforms. Use `bundle lock` on its
own if you only need to resolve/write the lockfile without installing gems.

Review and commit the resulting `Gemfile.lock` together with any Gemfile changes.
GitHub Pages pins several versions, so matching its supported bundle can retain
older versions than standalone Jekyll. See [GitHub Pages dependencies](https://pages.github.com/versions/).
Keep `.ruby-version`, the Gemfile's Ruby requirement, and the lockfile's Bundler
version consistent when updating the runtime. Verify the build and preview
before committing.

## GitHub Pages deployment

The publishing source is the **root directory of the `gh-pages` branch**.
Committing and pushing source changes to that branch triggers GitHub's native
Jekyll build and the **pages build and deployment** workflow. GitHub generates
the output and publishes it at <https://mikelsagardia.io>; do not commit `_site/`
or `vendor/`. Local edits and local servers do not deploy anything.

Check deployment results under the repository's
[Actions tab](https://github.com/mxagar/mxagar.github.io/actions).

## More on Jekyll

Check my guide: [https://github.com/mxagar/jekyll_web_guide](https://github.com/mxagar/jekyll_web_guide).

Mikel Sagardia, 2022.  
No guarantees.
