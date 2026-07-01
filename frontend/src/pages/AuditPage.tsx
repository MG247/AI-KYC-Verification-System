import { Braces, History, Loader2, Search } from 'lucide-react'
import { useState } from 'react'
import type { FormEvent } from 'react'
import { requestJson } from '../api/client'
import { InlineError, JsonBlock, PanelHeader } from '../components/common'

export function AuditPage({ latestRequestId }: { latestRequestId: string }) {
  const [auditId, setAuditId] = useState('')
  const [auditResult, setAuditResult] = useState<unknown>(null)
  const [auditError, setAuditError] = useState<string | null>(null)
  const [isAuditLoading, setIsAuditLoading] = useState(false)

  async function handleAuditSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setAuditError(null)
    setIsAuditLoading(true)

    try {
      const idToFetch = auditId.trim() || latestRequestId
      if (!idToFetch) {
        throw new Error('Enter a request id.')
      }
      setAuditResult(await requestJson<unknown>(`/audit/${encodeURIComponent(idToFetch)}`))
      setAuditId(idToFetch)
    } catch (error) {
      setAuditError(error instanceof Error ? error.message : 'Audit lookup failed.')
    } finally {
      setIsAuditLoading(false)
    }
  }

  return (
    <section className="content-grid">
      <form className="panel form-panel" onSubmit={handleAuditSubmit}>
        <PanelHeader icon={<History size={20} />} title="Audit Lookup" eyebrow="GET /audit/{request_id}" />
        <label className="field">
          <span>Request ID</span>
          <input value={auditId} onChange={(event) => setAuditId(event.target.value)} placeholder={latestRequestId || undefined} />
        </label>
        {auditError && <InlineError message={auditError} />}
        <button className="primary-button" type="submit" disabled={isAuditLoading}>
          {isAuditLoading ? <Loader2 className="spin" size={17} /> : <Search size={17} />}
          Fetch Audit
        </button>
      </form>

      <section className="panel results-panel">
        <PanelHeader icon={<Braces size={20} />} title="Audit Record" eyebrow={auditId || latestRequestId || 'No request selected'} />
        <JsonBlock data={auditResult} emptyText="Audit record will appear here." />
      </section>
    </section>
  )
}
