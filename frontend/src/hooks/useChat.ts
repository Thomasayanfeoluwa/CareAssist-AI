import { useCallback, useState } from 'react'
import { chat, type ChatResponse, type Source } from '../lib/api'

export type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources: Source[]
  model?: string
  requestId?: string
  error?: string
}

let counter = 0
function nextId(): string {
  counter += 1
  return `m${Date.now()}-${counter}`
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isSending, setIsSending] = useState(false)

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed || isSending) return

      const userMsg: Message = {
        id: nextId(),
        role: 'user',
        content: trimmed,
        sources: [],
      }
      setMessages((prev) => [...prev, userMsg])
      setIsSending(true)

      try {
        const resp: ChatResponse = await chat(trimmed)
        const assistantMsg: Message = {
          id: nextId(),
          role: 'assistant',
          content: resp.reply,
          sources: resp.sources,
          model: resp.model,
          requestId: resp.request_id,
        }
        setMessages((prev) => [...prev, assistantMsg])
      } catch (err) {
        const errorMsg: Message = {
          id: nextId(),
          role: 'assistant',
          content: '',
          sources: [],
          error:
            err instanceof Error ? err.message : String(err),
        }
        setMessages((prev) => [...prev, errorMsg])
      } finally {
        setIsSending(false)
      }
    },
    [isSending],
  )

  const clear = useCallback(() => {
    setMessages([])
  }, [])

  return { messages, isSending, send, clear }
}