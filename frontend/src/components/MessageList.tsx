import { useEffect, useRef } from 'react'
import type { Message } from '../hooks/useChat'
import { MessageBubble } from './MessageBubble'

export function MessageList({
  messages,
  isSending,
}: {
  messages: Message[]
  isSending: boolean
}) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages.length, isSending])

  if (messages.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6">
        <div className="max-w-md rounded-2xl border border-blue-100 bg-white/70 p-8 text-center shadow-sm">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-blue-50 ring-8 ring-blue-50/50">
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              className="h-8 w-8 text-blue-600"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M3 12h3l2-7 4 14 2-7h7" />
            </svg>
          </div>
          <h2 className="mt-5 font-serif text-2xl font-semibold tracking-tight text-slate-900">
            CareAssist
          </h2>
          <p className="mt-2 text-[15px] leading-relaxed text-slate-600">
            Ask a health question. CareAssist cites exact pages from a medical
            textbook and searches the web when the book does not cover a topic.
          </p>
          <p className="mt-4 text-xs text-slate-500">
            Educational information only · Not for diagnosis or treatment
          </p>
        </div>
      </div>
    )
  }

  return (
    <div
      className="flex flex-col gap-5 px-5 py-8"
      role="log"
      aria-live="polite"
      aria-relevant="additions"
      aria-label="Conversation"
    >
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
      {isSending && (
        <div className="flex justify-start" role="status">
          <div className="rounded-2xl rounded-bl-md border border-slate-200 bg-white px-5 py-3.5 text-sm text-slate-500 shadow-sm">
            <span className="inline-flex items-center gap-2">
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500" />
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500 [animation-delay:150ms]" />
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500 [animation-delay:300ms]" />
              <span className="ml-1">Thinking…</span>
            </span>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}