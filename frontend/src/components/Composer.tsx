import { useState } from 'react'

export function Composer({
  onSend,
  isSending,
}: {
  onSend: (text: string) => void
  isSending: boolean
}) {
  const [value, setValue] = useState('')

  const canSend = value.trim().length > 0 && !isSending

  const submit = () => {
    if (!canSend) return
    onSend(value)
    setValue('')
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t border-slate-200 bg-white px-4 py-3">
      <div className="flex items-end gap-2">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          placeholder="Ask a health question… (Enter to send, Shift+Enter for newline)"
          className="flex-1 resize-none rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-100"
          aria-label="Message"
          disabled={isSending}
        />
        <button
          type="button"
          onClick={submit}
          disabled={!canSend}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300"
        >
          {isSending ? 'Sending…' : 'Send'}
        </button>
      </div>
      <div className="mt-1 text-xs text-slate-400">
        {value.length}/2000 characters
      </div>
    </div>
  )
}