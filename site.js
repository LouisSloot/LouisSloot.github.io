/* ============================================================================
   site.js — pours the text from content.js into the page skeletons,
   and handles the light/dark theme toggle.

   You shouldn't need to touch this file to edit the site's text;
   that all lives in content.js.
   ============================================================================ */

(function () {
  "use strict";

  var C = window.SITE_CONTENT;
  if (!C) return;

  /* Post pages live one directory down, so links back up need a prefix. */
  var inPosts = /\/posts\//.test(location.pathname);
  var ROOT = inPosts ? "../" : "";

  /* ------------------------------- Theme -------------------------------- */
  /* The initial theme is set by a tiny inline script in each page's <head>
     (so there's no flash); this just wires up the toggle button. */

  function currentTheme() {
    return document.documentElement.getAttribute("data-theme") === "light"
      ? "light"
      : "dark";
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("theme", theme);
    } catch (e) {
      /* private browsing — the toggle still works, it just won't persist */
    }
    updateToggleButton();
  }

  function updateToggleButton() {
    var btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    var dark = currentTheme() === "dark";
    btn.textContent = dark ? "☀" : "☾";
    btn.setAttribute(
      "aria-label",
      dark ? "Switch to light mode" : "Switch to dark mode"
    );
    btn.title = btn.getAttribute("aria-label");
  }

  /* --------------------------- Header / footer --------------------------- */

  function renderHeader() {
    var mount = document.getElementById("site-header");
    if (!mount) return;

    var here = location.pathname.split("/").pop() || "index.html";
    if (inPosts) here = "blog.html"; /* highlight Blog while reading a post */

    var links = C.nav
      .map(function (item) {
        var active = item.href === here ? ' class="active"' : "";
        return "<a" + active + ' href="' + ROOT + item.href + '">' + item.label + "</a>";
      })
      .join("");

    mount.innerHTML =
      '<header class="container site-top">' +
      '<a class="site-name" href="' + ROOT + 'index.html">' + C.name + "</a>" +
      '<nav class="site-nav">' +
      links +
      '<button class="theme-toggle" type="button"></button>' +
      "</nav></header>";

    mount.querySelector(".theme-toggle").addEventListener("click", function () {
      setTheme(currentTheme() === "dark" ? "light" : "dark");
    });
    updateToggleButton();
  }

  function renderFooter() {
    var mount = document.getElementById("site-footer");
    if (!mount) return;

    var links = C.links
      .map(function (item) {
        var external = item.href.indexOf("http") === 0;
        return (
          '<a href="' + item.href + '"' +
          (external ? ' target="_blank" rel="noopener"' : "") +
          ">" + item.label + "</a>"
        );
      })
      .join('<span class="sep">·</span>');

    mount.innerHTML =
      '<footer class="site-footer"><div class="container footer-inner">' +
      "<span>" + C.name + "</span>" +
      '<span class="footer-links">' + links + "</span>" +
      "</div></footer>";
  }

  /* ----------------------------- Page content ---------------------------- */

  function renderHome() {
    var mount = document.getElementById("about");
    if (!mount) return;

    mount.innerHTML =
      "<h1>" + C.home.greeting + "</h1>" +
      C.home.bio.map(function (p) { return "<p>" + p + "</p>"; }).join("");

    var photo = document.getElementById("portrait");
    if (photo) {
      photo.src = C.home.photo;
      photo.alt = C.home.photoAlt;
    }
  }

  function renderProjects() {
    var mount = document.getElementById("project-list");
    if (!mount) return;

    mount.innerHTML = C.projects
      .map(function (p) {
        var html = '<article class="entry reveal">';
        html +=
          '<div class="entry-head"><h2 class="entry-title">' + p.title +
          '</h2><span class="entry-date">' + p.date + "</span></div>";
        if (p.video) {
          html +=
            '<video class="entry-video" autoplay loop muted playsinline preload="metadata">' +
            '<source src="' + p.video + '" type="video/mp4" /></video>';
        }
        html += "<p>" + p.desc + "</p>";
        if (p.note) html += '<p class="entry-note">' + p.note + "</p>";
        if (p.demo) {
          html +=
            '<a class="btn" href="' + p.demo + '" target="_blank" rel="noopener">Demo</a>';
        }
        return html + "</article>";
      })
      .join("");
  }

  function renderResearch() {
    var mount = document.getElementById("research-list");
    if (!mount) return;

    mount.innerHTML = C.research
      .map(function (r) {
        var html = '<article class="entry reveal">';
        html +=
          '<div class="entry-head"><h2 class="entry-title">' + r.title +
          '</h2><span class="entry-date">' + r.date + "</span></div>";
        html += "<p>" + r.desc + "</p>";
        if (r.image) {
          html +=
            '<figure class="entry-figure"><img src="' + r.image +
            '" alt="' + (r.imageAlt || r.title) + '" loading="lazy" /></figure>';
        }
        var links = "";
        if (r.paper) {
          links += '<a class="btn" href="' + r.paper + '" target="_blank" rel="noopener">Paper</a>';
        }
        if (r.poster) {
          links += '<a class="btn" href="' + r.poster + '" target="_blank" rel="noopener">Poster</a>';
        }
        if (links) html += '<div class="entry-links">' + links + "</div>";
        return html + "</article>";
      })
      .join("");
  }

  function renderBlog() {
    var mount = document.getElementById("blog-list");
    if (!mount) return;

    mount.innerHTML = C.blog
      .map(function (post) {
        return (
          '<article class="post-item reveal">' +
          '<h2><a href="' + post.href + '">' + post.title + "</a></h2>" +
          '<p class="meta">' + post.date + '<span class="sep">·</span>' + post.tag + "</p>" +
          '<p class="blurb">' + post.blurb + "</p>" +
          "</article>"
        );
      })
      .join("");
  }

  function renderConstruction() {
    var mount = document.getElementById("construction");
    if (!mount) return;

    var u = C.underConstruction;
    mount.innerHTML =
      '<div class="construction-emoji">' + u.emoji + "</div>" +
      "<h1>" + u.title + "</h1>" +
      '<p class="construction-subtitle">' + u.subtitle + "</p>" +
      '<div class="progress"><div class="progress-fill" style="width:' + u.progress + '%"></div></div>' +
      '<p class="construction-status">Progress: ' + u.progress + "%</p>" +
      u.statusLines
        .map(function (line) { return '<p class="construction-line">• ' + line + "</p>"; })
        .join("") +
      '<p class="construction-back"><a class="btn" href="index.html">' + u.backLabel + "</a></p>";
  }

  /* ---------------------------- Contact form ----------------------------- */
  /* Submits via FormSubmit (formsubmit.co) so a static site can email you with
     no backend. The target address is C.email; the first real submission
     triggers a one-time activation email you must confirm. See README. */

  function renderContact() {
    var mount = document.getElementById("contact");
    if (!mount || !C.contact) return;
    var k = C.contact;

    mount.innerHTML =
      '<h2 class="contact-heading">' + k.heading + "</h2>" +
      (k.intro ? '<p class="contact-intro">' + k.intro + "</p>" : "") +
      '<form class="contact-form" novalidate>' +
      '<div class="field-row">' +
      '<input name="name" type="text" placeholder="' + k.namePlaceholder + '" required autocomplete="name" />' +
      '<input name="email" type="email" placeholder="' + k.emailPlaceholder + '" required autocomplete="email" />' +
      "</div>" +
      '<textarea name="message" rows="5" placeholder="' + k.messagePlaceholder + '" required></textarea>' +
      /* Honeypot: humans leave it empty, bots fill it — then we drop the submit. */
      '<input type="text" name="_honey" class="hp" tabindex="-1" autocomplete="off" aria-hidden="true" />' +
      '<div class="contact-actions">' +
      '<button class="btn btn-solid" type="submit">' + k.send + "</button>" +
      '<span class="contact-status" role="status" aria-live="polite"></span>' +
      "</div></form>";

    var form = mount.querySelector("form");
    var status = mount.querySelector(".contact-status");
    var button = mount.querySelector("button");
    var endpoint = "https://formsubmit.co/ajax/" + encodeURIComponent(C.email);

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var data = new FormData(form);
      if (data.get("_honey")) return; /* bot */
      if (!form.checkValidity()) {
        form.reportValidity();
        return;
      }

      status.className = "contact-status";
      status.textContent = k.sending;
      button.disabled = true;

      fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({
          name: data.get("name"),
          email: data.get("email"),
          message: data.get("message"),
          _subject: k.subject,
          _template: "table",
        }),
      })
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function () {
          form.reset();
          status.className = "contact-status ok";
          status.textContent = k.success;
        })
        .catch(function () {
          status.className = "contact-status err";
          status.innerHTML = k.error;
        })
        .then(function () {
          button.disabled = false;
        });
    });
  }

  /* ------------------------------- Motion -------------------------------- */
  /* Subtle fade-up as list items scroll into view. Guarded by
     prefers-reduced-motion (CSS) with a no-IntersectionObserver fallback. */

  function setupReveal() {
    var els = [].slice.call(document.querySelectorAll(".reveal"));
    if (!els.length) return;

    var reduce =
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce || !("IntersectionObserver" in window)) {
      els.forEach(function (el) { el.classList.add("in"); });
      return;
    }

    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("in");
            io.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 }
    );
    els.forEach(function (el) { io.observe(el); });
  }

  /* Thin progress bar showing how far through a blog post you've read. */
  function setupReadingProgress() {
    if (!document.querySelector(".post")) return;

    var bar = document.createElement("div");
    bar.className = "progress-reading";
    document.body.appendChild(bar);

    var ticking = false;
    function update() {
      var doc = document.documentElement;
      var max = doc.scrollHeight - doc.clientHeight;
      var y = window.scrollY || doc.scrollTop;
      bar.style.width = (max > 0 ? (y / max) * 100 : 0) + "%";
      ticking = false;
    }
    window.addEventListener(
      "scroll",
      function () {
        if (!ticking) {
          window.requestAnimationFrame(update);
          ticking = true;
        }
      },
      { passive: true }
    );
    update();
  }

  renderHeader();
  renderFooter();
  renderHome();
  renderProjects();
  renderResearch();
  renderBlog();
  renderContact();
  renderConstruction();
  setupReveal();
  setupReadingProgress();
})();
