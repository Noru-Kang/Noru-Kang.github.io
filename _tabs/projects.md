---
icon: fas fa-folder-open
order: 6
title: Projects
permalink: /projects/
---

{% assign prioritized_projects = site.projects | where_exp: "item", "item.priority != nil" | sort: "priority" %}
{% assign unprioritized_projects = site.projects | where_exp: "item", "item.priority == nil" | sort: "title" %}
{% assign sorted_projects = prioritized_projects | concat: unprioritized_projects %}

{% if sorted_projects.size > 0 %}
<div class="projects-list">
  {% for project in sorted_projects %}
    <article class="project-item">
      {% if project.priority %}
        <p class="project-priority">Priority {{ project.priority }}</p>
      {% endif %}

      <h2 class="project-title">
        <a href="{{ project.url | relative_url }}">{{ project.title }}</a>
      </h2>

      {% if project.description %}
        <p class="project-description">{{ project.description }}</p>
      {% endif %}

      {% if project.tags %}
        <div class="project-tags">
          {% for tag in project.tags %}
            <span class="post-tag">{{ tag }}</span>
          {% endfor %}
        </div>
      {% endif %}

      <a class="project-link" href="{{ project.url | relative_url }}">Read project</a>
    </article>
  {% endfor %}
</div>
{% else %}
프로젝트가 아직 없습니다.
{% endif %}
