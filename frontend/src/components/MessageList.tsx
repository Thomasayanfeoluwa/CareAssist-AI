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
      <div className="flex h-full items-center justify-center px-6 text-center">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">CareAssist AI</h2>
          <p className="mt-2 text-sm text-slate-600">
            Ask me a health question. I will cite the medical book and web sources
            where I can, and I will say so when I do not have enough information.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div
      className="flex flex-col gap-4 px-4 py-6"
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
          <div className="rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-2 text-sm text-slate-500">
            <span className="inline-flex items-center gap-2">
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400" />
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400 [animation-delay:150ms]" />
              <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400 [animation-delay:300ms]" />
              <span className="ml-1">Thinking…</span>
            </span>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}