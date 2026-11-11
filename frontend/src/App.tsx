import { useChat } from './hooks/useChat'
import { MessageList } from './components/MessageList'
import { Composer } from './components/Composer'
import { Disclaimer } from './components/Disclaimer'
import { HealthIndicator } from './components/HealthIndicator'

export default function App() {
  const { messages, isSending, send, clear } = useChat()

  return (
    <div className="flex h-screen flex-col">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-2xl items-center justify-between px-6 py-3.5">
          <div className="flex items-center gap-3">
            <div
              className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600"
              aria-hidden="true"
            >
              <svg viewBox="0 0 24 24" className="h-4.5 w-4.5 text-white" fill="currentColor">
                <path d="M10 4h4v4h4v4h-4v4h-4v-4H6V8h4z" />
              </svg>
            </div>
            <div>
              <h1 className="font-serif text-lg font-semibold leading-tight tracking-tight text-slate-900">
                CareAssist
              </h1>
              <p className="text-[10.5px] font-medium uppercase tracking-[0.1em] text-slate-500">
                Educational Health Information
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <HealthIndicator />
            {messages.length > 0 && (
              <button
                type="button"
                onClick={clear}
                className="text-xs font-medium text-slate-500 transition-colors hover:text-blue-600"
              >
                New chat
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-hidden">
        <div className="mx-auto flex h-full max-w-2xl flex-col">
          <div className="flex-1 overflow-y-auto">
            <MessageList messages={messages} isSending={isSending} />
          </div>
          <Composer onSend={send} isSending={isSending} />
          <Disclaimer />
        </div>
      </main>
    </div>
  )
}