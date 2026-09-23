import { useEffect, useRef } from 'react'
import type { Message } from '../types'
import MessageBubble from './MessageBubble'
import { Sparkles } from 'lucide-react'

interface Props {
  messages: Message[]
  streaming: boolean
  suggestedQuestions: string[]
  onSelectQuestion: (question: string) => void
}

export default function ChatWindow({
  messages,
  streaming,
  suggestedQuestions,
  onSelectQuestion,
}: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streaming])

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin px-6 py-6 bg-slate-50">
      <div className="max-w-4xl mx-auto space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-slate-400 mt-32">
            <Sparkles className="w-12 h-12 mb-3" />
            <p className="text-base">开始一段对话吧 ✨</p>
            <p className="text-xs mt-1">先在上方上传文档，再向我提问</p>
            {suggestedQuestions.length > 0 && (
              <div className="mt-6 w-full max-w-2xl rounded-2xl border border-primary-100 bg-white p-4 shadow-sm">
                <p className="mb-3 text-sm font-medium text-slate-700">
                  你可以从这些问题开始
                </p>
                <div className="flex flex-col gap-2">
                  {suggestedQuestions.map((question) => (
                    <button
                      key={question}
                      onClick={() => onSelectQuestion(question)}
                      disabled={streaming}
                      className="rounded-lg border border-slate-200 px-3 py-2 text-left text-sm text-slate-700 hover:border-primary-300 hover:bg-primary-50 disabled:opacity-50"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {messages.map((m) => (
          <MessageBubble key={m.id} msg={m} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
