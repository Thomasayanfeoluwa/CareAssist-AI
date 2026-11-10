import { useChat } from './hooks/useChat'
import { MessageList } from './components/MessageList'
import { Composer } from './components/Composer'
import { Disclaimer } from './components/Disclaimer'
import { HealthStatus } from './components/HealthStatus'

export default function App() {
  const { messages, isSending, send, clear } = useChat()

  return (
    <div className="flex h-screen flex-col bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-2xl items-center justify-between px-4 py-3">
          <div>
            <h1 className="text-base font-semibold text-slate-900">CareAssist AI</h1>
            <p className="text-xs text-slate-500">
              Educational health-information assistant. Closed pilot.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <HealthStatus />
            {messages.length > 0 && (
              <button
                type="button"
                onClick={clear}
                className="rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:bg-slate-100"
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