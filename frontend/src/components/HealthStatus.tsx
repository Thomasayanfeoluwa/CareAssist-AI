import { useEffect, useState } from 'react'
import { fetchHealth, type Health } from '../lib/api'

export function HealthStatus() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchHealth()
      .then((h) => {
        if (!cancelled) setHealth(h)
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  if (error) {
    return (
      <p className="text-sm text-red-600" role="alert">
        API unreachable: {error}
      </p>
    )
  }
  if (!health) {
    return <p className="text-sm text-slate-500">Checking API...</p>
  }
  return (
    <p className="text-sm text-emerald-700">
      API status: {health.status} (v{health.version})
    </p>
  )
}