# LouisSloot.github.io

My personal website. Plain HTML/CSS/JS, no build step — GitHub Pages serves it as-is.

## Editing the site's text

**All hand-written text lives in one file: [`content.js`](content.js).**
Bio, project descriptions, research entries, the blog index, nav/footer links —
edit the string there, save, refresh. The HTML pages are just skeletons that
`site.js` fills in from `content.js`.

The one exception is full blog posts, which live as pages in `posts/`
(long-form writing with headings and images is easier to manage that way).

## Adding a blog post

1. Copy an existing file in `posts/` (e.g. `posts/thoughts-on-love.html`) and
   write the post inside its `<div class="post-content">`.
2. Add an entry for it to the `blog` list in `content.js` — that's what puts it
   on the blog page.

## Contact form

The home page has a contact form that emails you with no backend, via
[FormSubmit](https://formsubmit.co). Messages go to the `email` address set at
the top of `content.js`; the form's wording lives in the `contact` block there.

**One-time setup:** the first time anyone sends a message, FormSubmit emails
that address a confirmation link. Click it once to activate delivery — after
that, messages arrive automatically. (Easiest way to trigger it: send yourself
a test message from the live site.) To use a different inbox, change `email` in
`content.js` and confirm again.

## Structure

- `content.js` — **all the text** (edit this one)
- `site.js` — renders `content.js` into the pages + light/dark theme toggle
- `style.css` — one stylesheet; colors are variables at the top (dark is the
  default theme, `[data-theme="light"]` overrides)
- `index.html`, `projects.html`, `research.html`, `blog.html`, `dance.html` — page skeletons
- `posts/` — blog posts
- `visuals/`, `snippets/` — images and project demo videos
- `drafts/` — plain-text drafts of blog posts (not published)

## Previewing locally

```
python3 -m http.server
```

then open <http://localhost:8000>. (Opening `index.html` directly in a browser
works too.)
