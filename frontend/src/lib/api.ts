import { z } from 'zod'

const HealthSchema = z.object({
  status: z.string(),
  version: z.string(),
})

const SourceSchema = z.object({
  type: z.enum(['pdf', 'web']),
  title: z.string(),
  page: z.number().int().nullable().optional(),
  link: z.string().nullable().optional(),
  snippet: z.string().optional().default(''),
})

const ChatResponseSchema = z.object({
  reply: z.string(),
  model: z.string(),
  sources: z.array(SourceSchema),
  request_id: z.string(),
})

export type Health = z.infer<typeof HealthSchema>
export type Source = z.infer<typeof SourceSchema>
export type ChatResponse = z.infer<typeof ChatResponseSchema>

const BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

const PILOT_TOKEN: string =
  import.meta.env.VITE_PILOT_AUTH_TOKEN ?? ''

export async function fetchHealth(): Promise<Health> {
  const res = await fetch(`${BASE_URL}/healthz`, { method: 'GET' })
  if (!res.ok) {
    throw new Error(`healthz failed: ${res.status} ${res.statusText}`)
  }
  return HealthSchema.parse(await res.json())
}

export async function chat(message: string): Promise<ChatResponse> {
  if (!PILOT_TOKEN) {
    throw new Error(
      'VITE_PILOT_AUTH_TOKEN is not set — add it to frontend/.env.local',
    )
  }
  const res = await fetch(`${BASE_URL}/api/chat`, {
    method: 'POST',
    credentials: 'include',          // send session cookie
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${PILOT_TOKEN}`,
    },
    body: JSON.stringify({ message }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`chat failed: ${res.status} ${text.slice(0, 200)}`)
  }
  return ChatResponseSchema.parse(await res.json())
}