---
title: "我的项目作品集"
permalink: /portfolio/
layout: archive
collection: portfolio
---

## 项目列表  
{% for post in site.portfolio %}  
  <div class="post-preview">  
    <a href="{{ post.url | relative_url }}">  
      <h2 class="post-title">{{ post.title }}</h2>  
      <h3 class="post-subtitle">{{ post.excerpt | strip_html }}</h3>  
    </a>  
    <p class="post-meta">发布时间: {{ post.date | date: "%Y-%m-%d" }}</p>  
  </div>  
  <hr>  
{% endfor %}  
