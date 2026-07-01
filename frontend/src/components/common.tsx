import { AlertTriangle, BadgeCheck, Braces } from 'lucide-react'
import type { ReactNode } from 'react'
import { decisionTone, formatLabel, formatValue, scorePercent } from '../utils/format'

export function PanelHeader({ icon, title, eyebrow }: { icon: ReactNode; title: string; eyebrow: string }) {
  return (
    <div className="panel-header">
      <span className="panel-icon">{icon}</span>
      <div>
        <span>{eyebrow}</span>
        <h2>{title}</h2>
      </div>
    </div>
  )
}

export function StatusBadge({ value }: { value: string }) {
  const tone = decisionTone(value)
  return <span className={`badge ${tone}`}>{formatLabel(value)}</span>
}

export function ScoreCard({
  label,
  value,
  detail,
  tone,
  invert = false,
}: {
  label: string
  value: number
  detail: string
  tone: string
  invert?: boolean
}) {
  const percent = scorePercent(value)
  const displayPercent = invert ? 100 - percent : percent

  return (
    <article className="score-card">
      <div>
        <span>{label}</span>
        <strong>{Math.round(percent)}%</strong>
        <small>{formatLabel(detail)}</small>
      </div>
      <div className="progress-track" aria-hidden="true">
        <span className={tone} style={{ width: `${displayPercent}%` }} />
      </div>
    </article>
  )
}

export function KeyValueGrid({ data }: { data: Record<string, unknown> }) {
  const entries = Object.entries(data).filter(([, value]) => value !== null && value !== undefined && value !== '')
  if (!entries.length) {
    return <EmptyState icon={<Braces size={22} />} text="No document fields returned." />
  }
  return (
    <dl className="key-value-grid">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{formatLabel(key)}</dt>
          <dd>{formatValue(value)}</dd>
        </div>
      ))}
    </dl>
  )
}

export function ReasonList({ items, fallback }: { items: string[]; fallback: string }) {
  if (!items.length) {
    return <EmptyState icon={<BadgeCheck size={22} />} text={fallback} />
  }
  return (
    <ul className="reason-list">
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  )
}

export function FeedbackList({ items }: { items: Array<Record<string, unknown>> }) {
  if (!items.length) {
    return <EmptyState icon={<BadgeCheck size={22} />} text="No document feedback returned." />
  }
  return (
    <div className="feedback-list">
      {items.map((item, index) => (
        <article key={`${String(item.category ?? 'feedback')}-${index}`}>
          <StatusBadge value={String(item.severity ?? 'INFO')} />
          <div>
            <strong>{formatLabel(String(item.category ?? 'Feedback'))}</strong>
            <p>{String(item.message ?? 'No message returned.')}</p>
          </div>
        </article>
      ))}
    </div>
  )
}

export function JsonBlock({ data, emptyText = 'No data returned.' }: { data: unknown; emptyText?: string }) {
  if (data === null || data === undefined) {
    return <EmptyState icon={<Braces size={22} />} text={emptyText} />
  }
  return <pre className="json-block">{JSON.stringify(data, null, 2)}</pre>
}

export function EmptyState({ icon, text }: { icon: ReactNode; text: string }) {
  return (
    <div className="empty-state">
      {icon}
      <span>{text}</span>
    </div>
  )
}

export function InlineError({ message }: { message: string }) {
  return (
    <div className="inline-error" role="alert">
      <AlertTriangle size={16} />
      <span>{message}</span>
    </div>
  )
}
