import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import type { Message } from '../hooks/useChat'
import { SourceChip } from './SourceChip'

export function MessageBubble({ message }: { message: Message }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap break-words rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2 text-white">
          {message.content}
        </div>
      </div>
    )
  }

  if (message.error) {
    return (
      <div className="flex justify-start">
        <div className="max-w-[80%] rounded-2xl rounded-bl-sm border border-red-200 bg-red-50 px-4 py-2 text-red-700">
          <div className="text-xs font-semibold uppercase">Error</div>
          <div className="mt-1 whitespace-pre-wrap break-words">{message.error}</div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-2xl rounded-bl-sm border border-slate-200 bg-white px-4 py-2 text-slate-900">
        <div className="prose prose-sm prose-slate max-w-none">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeSanitize]}
          >
            {message.content}
          </ReactMarkdown>
        </div>
        {message.sources.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5 border-t border-slate-100 pt-2">
            <span className="mr-1 text-xs text-slate-500">Sources:</span>
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