import { createElement } from 'react'
import { renderToString } from 'react-dom/server'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'

const cases = [
  {
    name: 'img-onerror',
    input: 'safe <img src=x onerror=alert(1)> text',
    forbidden: ['<img'],
  },
  {
    name: 'script-tag',
    input: 'before <script>alert(1)</script> after',
    forbidden: ['<script'],
  },
  {
    name: 'javascript-link',
    input: '[click me](javascript:alert(1))',
    forbidden: ['href="javascript:'],
  },
  {
    name: 'event-handler-attr',
    input: '<div onclick="steal()">hi</div>',
    forbidden: ['onclick='],
  },
  {
    name: 'safe-link-still-works',
    input: '[Mayo Clinic](https://mayoclinic.org)',
    required: ['href="https://mayoclinic.org"'],
  },
  {
    name: 'bold-markdown-still-works',
    input: '**important**',
    required: ['<strong>important</strong>'],
  },
]

let passed = 0
let failed = 0
for (const c of cases) {
  const html = renderToString(
    createElement(ReactMarkdown, {
      remarkPlugins: [remarkGfm],
      rehypePlugins: [rehypeSanitize],
      children: c.input,
    })
  )
  let ok = true
  for (const f of c.forbidden || []) {
    if (html.includes(f)) {
      console.log(`FAIL [${c.name}]: found forbidden "${f}"\n  html: ${html}`)
      ok = false
    }
  }
  for (const r of c.required || []) {
    if (!html.includes(r)) {
      console.log(`FAIL [${c.name}]: missing required "${r}"\n  html: ${html}`)
      ok = false
    }
  }
  if (ok) {
    console.log(`OK [${c.name}]`)
    passed++
  } else {
    failed++
  }
}

console.log(`\n${passed} passed, ${failed} failed`)
process.exit(failed > 0 ? 1 : 0)