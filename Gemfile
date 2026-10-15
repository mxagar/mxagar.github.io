# frozen_string_literal: true

source "https://rubygems.org"

ruby "~> 3.3.0"

# Match the native GitHub Pages builder, including Jekyll 3.10 and Minima 2.5.1.
gem "github-pages", "~> 232", group: :jekyll_plugins
gem "minima", "~> 2.5"

group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.17"
  gem "jemoji", "~> 0.13"
end

# Ruby no longer includes the HTTP server used by `jekyll serve`.
gem "webrick", "~> 1.9"

# Retry middleware for the GitHub metadata client.
gem "faraday-retry", "~> 2.3"
