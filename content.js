/* ============================================================================
   content.js — every piece of hand-written text on the site lives here.

   Want to reword something? It's almost certainly in this file.
   Edit the string, save, refresh the page. No build step.

   Notes:
   - Strings can contain HTML (<a>, <em>, ...) and are injected as-is.
   - The one exception: full blog posts live in posts/*.html, since long-form
     writing with headings/images is easier to manage as its own page.
     The "blog" list below only controls what shows up on blog.html.
   - To add a blog post: copy an existing file in posts/, write in it,
     then add an entry to the "blog" list below.
   ============================================================================ */

window.SITE_CONTENT = {
  name: "Louis Sloot",
  email: "lsloot@andrew.cmu.edu",

  /* Header navigation (the site name on the left always links home). */
  nav: [
    { label: "Projects", href: "projects.html" },
    { label: "Research", href: "research.html" },
    { label: "Blog", href: "blog.html" },
  ],

  /* Shown in the footer on every page. */
  links: [
    { label: "GitHub", href: "https://github.com/LouisSloot" },
    { label: "LinkedIn", href: "https://www.linkedin.com/in/louis-sloot-26bb60324" },
    {
      label: "Resume",
      href: "https://drive.google.com/file/d/1vEYcECdl1wo1-BuhM1AByEO_yYLtIXNS/view?usp=sharing",
    },
    { label: "Contact", href: "index.html#contact" },
  ],

  /* ------------------------- Contact form (home) ------------------------- */
  /* The form on the home page emails you via FormSubmit — messages go to the
     "email" address above. No server needed. IMPORTANT: the very first message
     sent triggers a one-time confirmation email from FormSubmit to that
     address; click the link in it once to activate delivery. */
  contact: {
    heading: "Get in touch",
    intro: "Let's talk about something you love.",
    namePlaceholder: "Your name",
    emailPlaceholder: "Your email",
    messagePlaceholder: "Your message",
    send: "Send",
    sending: "Sending…",
    subject: "New message from your website",
    success: "Thanks! Your message is on its way — I'll get back to you soon.",
    /* Shown if the send fails; may contain HTML. */
    error:
      'Hmm, that didn\'t go through. You can also email me at <a href="mailto:lsloot@andrew.cmu.edu">lsloot@andrew.cmu.edu</a>.',
  },

  /* ------------------------------ Home page ------------------------------ */
  home: {
    greeting: 'Hey, I\'m <span class="accent">Louis</span>.',
    /* One string per paragraph. (Class year lives in the first one.) */
    bio: [
      "You look great today! And welcome to my website. I'm a rising junior at \
Carnegie Mellon University, majoring in \
Artificial Intelligence. I'm passionate about all things AI safety. I'd love to talk with you about that.",

      'I hail from St. Paul, MN. Growing up, I spanned the gap between arts and academics as best I could. From \
grades 1&ndash;11, I was a competitive dancer at a local studio \
<a href="dance.html">(what?! dance?!)</a>, and I took my senior year to pursue \
more professional performance opportunities in the local arts scene. In \
school, I loved just about everything thrown at me (but always a little extra \
with math); and when I built my own gaming PC in 8th grade, I began my descent \
into the wondrous world that is Computer Science. Today, my life is still defined by tech and dance.',
    ],
    photo: "visuals/Cropped_Sr_Pic.jpg",
    photoAlt: "A lovely photo of me! Albeit 3 years old.",
  },

  /* ----------------------------- Projects page --------------------------- */
  /* Listed top-to-bottom in this order. "video" is optional; "demo" is an
     optional external link; "note" is an optional muted line (e.g. WIP). */
  projects: [
    {
      title: "Selfstat",
      date: "July 2025 – present",
      note: "Demo coming soon",
      desc: "Selfstat automates stat-tracking for all basketball games--not just the NBA. A user \
uploads amateur footage of a basketball game, and it will generate the \
corresponding box score for each player.",
    },
    {
      title: "C0 Bytecode Interpreter",
      date: "April 2025",
      video: "snippets/Bytecode_Interpreter_Snippet.mp4",
      desc: "My implementation of a virtual machine for the C0 programming \
language for 15-122 @CMU. ",
    },
    {
      title: "Portfolio Website",
      date: "January 2025",
      video: "snippets/Portfolio_Website_Snippet.mp4",
      desc: "Woah... meta.",
    },
    {
      title: "The Fractal Factory",
      date: "December 2024",
      video: "snippets/Fractal_Factory_Snippet.mp4",
      demo: "https://drive.google.com/file/d/1F-tGEdFseoLUMFqDfaoqOApcItgGBfK2/view?usp=sharing",
      desc: "My final project for Intro CS @CMU. Fractals are awesome.",
    },
    {
      title: "Finger Ninja",
      date: "November 2024",
      video: "snippets/112_Ninja_Snippet.mp4",
      demo: "https://drive.google.com/file/d/1TF1QpsigkTDT6b76vKdtqdvWwOTlZkZk/view?usp=sharing",
      desc: "My first ever Hackathon! Look at me slice those digital fruit. A \
rite of passage here at CMU, my 1st Place Hack112 experience was awesome.",
    },
    {
      title: "HPE and Dance",
      date: "February 2024",
      video: "snippets/HL_IA_Snippet.mp4",
      demo: "https://youtu.be/nFM3FxSrJlU",
      desc: "This was my first CS project. In short, \
I built a software to aid my high school's dance teacher: it can analyze his students' dancing and provide accurate feedback on \
necessary improvements.",
    },
  ],

  /* ----------------------------- Research page --------------------------- */
  /* "paper" and "poster" are optional external links; "image" (with optional
     "imageAlt") is an optional figure shown under the description. */
  research: [
    {
      title: "Supervised Program for Alignment Research",
      date: "September 2025 – present",
      paper: "https://arxiv.org/abs/2603.02297",
      poster: "https://drive.google.com/file/d/1leUXPjdRgSE_Jlm0v84WXXo1dvDAgaHc/view?usp=sharing",
      desc: "Frontier LLMs are evaluated on a large, standardized benchmark \
suite. In this suite, however, exists a crucial gap with respect to \
cybersecurity tasks, especially in mission-critical systems. As a current \
Research Fellow with SPAR, I work on the development of ZeroDayBench, a \
cybersecurity benchmark that measures a frontier model's ability to detect and \
patch vulnerabilities in pivotal environments. This includes identifying and \
replicating real-world vulnerabilities, then writing automated penetration \
tests and leveraging an MCP framework to evaluate agent patches. My team and \
I are preparing our findings for ICML 2026.",
    },
    {
      title: "Dr. Wei Wu's CompBio Lab",
      date: "February 2025 – August 2025",
      image: "visuals/bio_research.png",
      imageAlt:
        "UMAP projection of scRNA-seq cells, clustered and labeled into cell-type populations including CD4/CD8 T cells, monocytes, B cells, NK cells, and platelets.",
      desc: "With the ever-growing impact of large single cell RNA-sequenced \
(scRNA-seq) datasets, effective methods for working with data at scale become \
only more pertinent. In this computational biology lab at CMU, my work as an \
undergraduate research assistant centered around improving identification of \
rare cell types in scRNA-seq datasets, relying on unsupervised machine \
learning techniques. Leveraging R for algorithm design and data analysis, I \
developed software to reliably cluster various cell populations, including \
gamma delta T cells, which are invaluable for cancer immunotherapy \
treatments.",
    },
    {
      title: "OurCS @CMU",
      date: "October 2024",
      poster: "https://drive.google.com/file/d/1T5I0tRfMZhX7XGRcPNW2eHNPkiZpEqg0/view?usp=sharing",
      desc: "The OurCS research conference is an opportunity for undergrad \
students to make meaningful research contributions. In conjunction with two \
undergrad team members and working under a PhD Researcher in AI Optimization \
from Adobe, I trained, tested, and pruned a standard CNN in PyTorch to \
empirically locate optimal sparsity ratios that can balance performance and \
efficiency. We presented our findings to a panel of SCS faculty and \
conference personnel.",
    },
  ],

  /* ------------------------------- Blog index ---------------------------- */
  /* Listed top-to-bottom in this order. The actual posts live in posts/. */
  blog: [
    {
      title: "CMU Megathread",
      href: "posts/cmu-megathread.html",
      date: "Ongoing",
      tag: "CMU",
      blurb: "A reflection on my experience at Carnegie Mellon.",
    },
    {
      title: "Representation Engineering: A Top-Down Approach to AI Transparency",
      href: "posts/repe-paper-review.html",
      date: "January 2026",
      tag: "Paper Review",
      blurb: "An overview of Representation Engineering (RepE).",
    },
    {
      title: "Thoughts on <em>Thoughts on Love</em>",
      href: "posts/thoughts-on-love.html",
      date: "December 2025",
      tag: "Musings",
      blurb: "Reflections on Anne Lamott's <em>Somehow: Thoughts on Love</em>.",
    },
  ],

  /* --------------------------- dance.html (joke) ------------------------- */
  underConstruction: {
    emoji: "🚧",
    title: "Under Construction",
    subtitle: "Thank you for your patience.",
    progress: 23, /* percent */
    statusLines: [
      "Compiling motivation...",
      "Rationalizing procrastination",
      "ETA: Soonish™",
    ],
    backLabel: "← Back to Safety",
  },
};
