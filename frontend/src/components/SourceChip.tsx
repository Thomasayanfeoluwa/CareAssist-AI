import { useState } from 'react'
import * as Popover from '@radix-ui/react-popover'
import type { Source } from '../lib/api'

export function SourceChip({ source, index }: { source: Source; index: number }) {
  const [open, setOpen] = useState(false)
  const isPdf = source.type === 'pdf'
  const label = isPdf ? `PDF p.${source.page ?? '?'}` : `Web ${index + 1}`

  const enter = (e: React.PointerEvent) => {
    if (e.pointerType === 'mouse') setOpen(true)
  }
  const leave = (e: React.PointerEvent) => {
    if (e.pointerType === 'mouse') setOpen(false)
  }

  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button
          type="button"
          className="inline-flex items-center rounded-md border border-blue-200 bg-blue-50 px-2 py-0.5 text-[11px] font-medium text-blue-700 transition-colors hover:border-blue-300 hover:bg-blue-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
          aria-haspopup="dialog"
          aria-label={`View source: ${source.title}${source.page ? `, page ${source.page}` : ''}`}
          onPointerEnter={enter}
          onPointerLeave={leave}
          onFocus={() => setOpen(true)}
          onBlur={(e) => {
            if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOpen(false)
          }}
        >
          {label}
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          className="z-50 w-80 rounded-lg border border-slate-200 bg-white p-4 text-xs shadow-lg"
          sideOffset={6}
          role="dialog"
          aria-label="Source details"
          onPointerEnter={enter}
          onPointerLeave={leave}
          onOpenAutoFocus={(e) => e.preventDefault()}
        >
          <div className="font-serif text-sm font-semibold text-slate-900">{source.title}</div>
          {isPdf && source.page != null && (
            <div className="mt-1 text-[11px] uppercase tracking-wider text-slate-500">
              Page {source.page}
            </div>
          )}
          {!isPdf && source.link && (
            <a
              href={source.link}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 block break-all text-blue-600 hover:underline"
            >
              {source.link}
            </a>
          )}
          {source.snippet && (
            <div className="mt-3 border-t border-slate-100 pt-3 text-[12px] leading-relaxed text-slate-600">
              {source.snippet}
            </div>
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  )
}