export function Disclaimer() {
  return (
    <div className="border-t border-blue-100 bg-blue-50/70 px-6 py-3">
      <div className="mx-auto flex max-w-2xl items-center justify-center gap-2 text-center text-[11px] leading-relaxed text-slate-600">
        <svg
          aria-hidden="true"
          viewBox="0 0 16 16"
          className="h-3.5 w-3.5 flex-shrink-0 text-blue-600"
          fill="currentColor"
        >
          <path d="M8 1.5 3 3.5v4.2c0 3.1 2.1 5.9 5 6.8 2.9-.9 5-3.7 5-6.8V3.5L8 1.5Z" />
        </svg>
        <span>
          Educational information only — not medical advice.
          <span className="hidden sm:inline">
            {' '}Always consult a qualified health professional for your specific situation.
          </span>
        </span>
      </div>
    </div>
  )
}