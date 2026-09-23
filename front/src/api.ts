import type { BrainInfo, SessionInfo } from './types'

const BASE = import.meta.env.VITE_API_BASE || ''
const API_KEY = import.meta.env.VITE_API_KEY || ''

function headers(extra: Record<string, string> = {}): HeadersInit {
  const h: Record<string, string> = { ...extra }
  if (API_KEY) h['X-API-Key'] = API_KEY
  return h
}

export async function createSession(brainName: string): Promise<SessionInfo> {
  const r = await fetch(`${BASE}/sessions`, {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ brain_name: brainName }),
  })
  if (!r.ok) throw new Error(`createSession ${r.status}`)
  return r.json()
}

export async function getSession(sid: string): Promise<SessionInfo> {
  const r = await fetch(`${BASE}/sessions/${sid}`, { headers: headers() })
  if (!r.ok) throw new Error(`getSession ${r.status}`)
  return r.json()
}

export async function deleteSession(sid: string): Promise<void> {
  await fetch(`${BASE}/sessions/${sid}`, {
    method: 'DELETE',
    headers: headers(),
  })
}

export async function uploadFiles(
  sid: string,
  files: File[],
): Promise<{ uploaded: string[]; total_chunks: number }> {
  const fd = new FormData()
  files.forEach((f) => fd.append('files', f))
  const r = await fetch(`${BASE}/upload?session_id=${sid}`, {
    method: 'POST',
    headers: headers(),
    body: fd,
  })
  if (!r.ok) throw new Error(`upload ${r.status}: ${await r.text()}`)
  return r.json()
}

export async function getSuggestedQuestions(
  sid: string,
  count = 5,
): Promise<string[]> {
  const r = await fetch(
    `${BASE}/sessions/${sid}/suggested-questions?count=${count}`,
    { headers: headers() },
  )
  if (!r.ok) throw new Error(`suggestedQuestions ${r.status}`)
  const data: { questions: string[] } = await r.json()
  return data.questions
}

export async function askStream(
  sid: string,
  question: string,
  onChunk: (text: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const r = await fetch(`${BASE}/ask_stream`, {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ session_id: sid, question, use_history: true }),
    signal,
  })
  if (!r.ok || !r.body) throw new Error(`ask_stream ${r.status}`)

  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    const lines = buf.split('\n\n')
    buf = lines.pop() || ''
    for (const line of lines) {
      const payload = line.replace(/^data:\s*/, '').trim()
      if (payload === '[DONE]') return
      if (payload.startsWith('[ERROR]')) throw new Error(payload)
      if (payload) onChunk(payload)
    }
  }
}

export async function getBrainInfo(sid: string): Promise<BrainInfo> {
  const r = await fetch(`${BASE}/info?session_id=${sid}`, { headers: headers() })
  if (!r.ok) throw new Error(`info ${r.status}`)
  return r.json()
}

export async function healthCheck(): Promise<{ status: string; redis: string }> {
  const r = await fetch(`${BASE}/health`)
  return r.json()
}
