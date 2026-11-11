import { useEffect, useState } from 'react'
import { fetchHealth } from '../lib/api'

export function HealthIndicator() {
  const [down, setDown] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetchHealth()
      .then(() => { if (!cancelled) setDown(false) })
      .catch(() => { if (!cancelled) setDown(true) })
    return () => { cancelled = true }
  }, [])

  if (!down) return null
  return (
    <span
      className="flex items-center gap-1.5 text-[11px] font-medium text-amber-700"
      title="Cannot reach the assistant service"
    >
      <span className="inline-block h-1.5 w-1.5 rounded-full bg-amber-500" />
      Connection issue
    </span>
  )
}