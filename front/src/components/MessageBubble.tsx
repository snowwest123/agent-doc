import type { Message } from '../types'
import { Bot, User } from 'lucide-react'

interface Props {
  msg: Message
}

export default function MessageBubble({ msg }: Props) {
  const isUser = msg.role === 'user'
  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-primary-500 flex items-center justify-center flex-shrink-0">
          <Bot className="w-5 h-5 text-white" />
        </div>
      )}
      <div
        className={`max-w-2xl px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? 'bg-primary-600 text-white'
            : 'bg-white border border-slate-200 text-slate-800'
        }`}
      >
        {msg.content}
        {msg.streaming && (
          <span className="inline-block w-2 h-4 ml-1 bg-slate-400 animate-pulse" />
        )}
      </div>
      {isUser && (
        <div className="w-8 h-8 rounded-full bg-slate-300 flex items-center justify-center flex-shrink-0">
          <User className="w-5 h-5 text-slate-700" />
        </div>
      )}
    </div>
  )
}