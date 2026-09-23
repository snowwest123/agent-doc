import { useRef, useState } from 'react'
import { askStream } from '../api'

export function useStream() {
  const [streaming, setStreaming] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  async function start(
    sid: string,
    question: string,
    onChunk: (text: string) => void,
  ): Promise<string> {
    setStreaming(true)
    abortRef.current = new AbortController()
    let full = ''
    try {
      await askStream(
        sid,
        question,
        (chunk) => {
          full += chunk
          onChunk(chunk)
        },
        abortRef.current.signal,
      )
    } finally {
      setStreaming(false)
    }
    return full
  }

  function stop() {
    abortRef.current?.abort()
    setStreaming(false)
  }

  return { streaming, start, stop }
}