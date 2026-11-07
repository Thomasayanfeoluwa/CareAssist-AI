import { HealthStatus } from './components/HealthStatus'

export default function App() {
  return (
    <main className="mx-auto flex min-h-full max-w-2xl flex-col items-center justify-center p-8">
      <h1 className="text-2xl font-semibold tracking-tight">CareAssist AI</h1>
      <p className="mt-2 text-sm text-slate-600">
        Educational health-information assistant. Closed pilot.
      </p>
      <div className="mt-6 rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
        <HealthStatus />
      </div>
    </main>
  )
}