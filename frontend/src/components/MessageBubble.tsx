import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import type { Message } from '../hooks/useChat'
import { SourceChip } from './SourceChip'

function stripMarkdown(md: string): string {
  return md
    .replace(/\[PDF: [^\]]+\]/g, '')
    .replace(/\[Web: [^\]]+\]/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_`#]/g, '')
    .replace(/^\s*[-•]\s*/gm, '')
    .replace(/\n{2,}/g, '. ')
    .trim()
}

function useSpeaking(): { speaking: boolean; speak: (t: string) => void; stop: () => void } {
  const [speaking, setSpeaking] = useState(false)
  useEffect(() => {
    const onEnd = () => setSpeaking(false)
    window.speechSynthesis.addEventListener('end', onEnd)
    return () => {
      window.speechSynthesis.removeEventListener('end', onEnd)
      window.speechSynthesis.cancel()
    }
  }, [])
  const speak = (text: string) => {
    window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(stripMarkdown(text))
    u.lang = 'en-US'
    u.rate = 1.0
    u.onend = () => setSpeaking(false)
    u.onerror = () => setSpeaking(false)
    setSpeaking(true)
    window.speechSynthesis.speak(u)
  }
  const stop = () => {
    window.speechSynthesis.cancel()
    setSpeaking(false)
  }
  return { speaking, speak, stop }
}

export function MessageBubble({ message }: { message: Message }) {
  const { speaking, speak, stop } = useSpeaking()

  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap break-words rounded-2xl rounded-br-md bg-blue-600 px-4 py-2.5 text-[15px] leading-relaxed text-white shadow-sm">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.error) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[80%] rounded-2xl rounded-bl-md border border-amber-200 bg-amber-50 px-4 py-2.5 text-amber-900">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-amber-700">
            Something went wrong
          </div>
          <div className="mt-1 whitespace-pre-wrap break-words text-sm">{message.error}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-2xl rounded-bl-md border border-slate-200 bg-white px-5 py-4 text-slate-900 shadow-sm">
        <div className="prose prose-sm prose-slate max-w-none">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>
            {message.content}
          </ReactMarkdown>
        </div>

        <div className="mt-3 flex items-center gap-2">
          <button
            type="button"
            onClick={() => (speaking ? stop() : speak(message.content))}
            className="rounded-md border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 transition-colors hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            aria-label={speaking ? 'Stop reading the answer' : 'Read this answer aloud'}
            aria-pressed={speaking}
          >
            {speaking ? 'Stop' : 'Read aloud'}
          </button>
        </div>

        {message.sources.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5 border-t border-slate-100 pt-3">
            <span className="mr-1 text-[11px] font-medium uppercase tracking-wider text-slate-400">
              Sources
            </span>
            {message.sources.map((s, i) => (
              <SourceChip
                key={`${i}-${s.type}-${s.title}-${s.page ?? ''}-${s.link ?? ''}`}
                source={s}
                index={i}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}