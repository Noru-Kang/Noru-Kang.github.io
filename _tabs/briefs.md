---
layout: portfolio
icon: fas fa-briefcase
order: 1
title: Briefs
permalink: /briefs/
---

## Briefs

프로젝트 요약 및 연구 브리프를 모아보는 페이지입니다.
아래 카드는 기본적으로 `projects` 카테고리 포스트를 자동으로 모아 보여줍니다.

{% assign sample_post = site.posts | first %}
{% if sample_post %}
예시 포스트 링크: [{{ sample_post.title }}]({{ sample_post.url | relative_url }})
{% endif %}
