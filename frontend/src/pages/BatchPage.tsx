import { ClipboardList, Gauge, Loader2 } from 'lucide-react'
import { useState } from 'react'
import type { FormEvent } from 'react'
import { requestJson } from '../api/client'
import { EmptyState, InlineError, JsonBlock, PanelHeader, StatusBadge } from '../components/common'
import { defaultBatchPayload } from '../data/defaults'
import type { BatchResponse } from '../types'
import { isRecord } from '../utils/format'

export function BatchPage({
  onRequestIdFound,
  onStatusRefresh,
}: {
  onRequestIdFound: (requestId: string) => void
  onStatusRefresh: () => Promise<void>
}) {
  const [batchJson, setBatchJson] = useState(JSON.stringify(defaultBatchPayload, null, 2))
  const [batchResult, setBatchResult] = useState<BatchResponse | null>(null)
  const [batchError, setBatchError] = useState<string | null>(null)
  const [isBatchRunning, setIsBatchRunning] = useState(false)

  async function handleBatchSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBatchError(null)
    setIsBatchRunning(true)

    try {
      const payload = JSON.parse(batchJson) as Record<string, unknown>
      const result = await requestJson<BatchResponse>('/batch-verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      setBatchResult(result)
      const firstRequest = result.results.find((item) => isRecord(item) && typeof item.request_id === 'string')
      if (isRecord(firstRequest) && typeof firstRequest.request_id === 'string') {
        onRequestIdFound(firstRequest.request_id)
      }
      await onStatusRefresh()
    } catch (error) {
      setBatchError(error instanceof Error ? error.message : 'Batch verification failed.')
    } finally {
      setIsBatchRunning(false)
    }
  }

  return (
    <section className="content-grid">
      <form className="panel form-panel" onSubmit={handleBatchSubmit}>
        <PanelHeader icon={<ClipboardList size={20} />} title="Batch Verification" eyebrow="POST /batch-verify" />
        <label className="field">
          <span>Batch JSON</span>
          <textarea value={batchJson} onChange={(event) => setBatchJson(event.target.value)} rows={18} />
        </label>
        {batchError && <InlineError message={batchError} />}
        <button className="primary-button" type="submit" disabled={isBatchRunning}>
          {isBatchRunning ? <Loader2 className="spin" size={17} /> : <ClipboardList size={17} />}
          Run Batch
        </button>
      </form>

      <section className="panel results-panel">
        <PanelHeader icon={<Gauge size={20} />} title="Batch Results" eyebrow={`${batchResult?.count ?? 0} documents`} />
        {batchResult?.results?.length ? (
          <div className="batch-list">
            {batchResult.results.map((result, index) => {
              const row = (isRecord(result) ? result : {}) as Record<string, unknown>
              const rowKey = String(row.request_id ?? row.document_path ?? index)
              const rowTitle = String(row.document_type_label ?? row.document_type ?? row.document_path ?? `Document ${index + 1}`)
              const rowDetail = String(row.request_id ?? row.error ?? 'Completed')
              const rowDecision = String(row.final_decision ?? (row.error ? 'ERROR' : 'DONE'))

              return (
                <article className="batch-item" key={`${index}-${rowKey}`}>
                  <div>
                    <strong>{rowTitle}</strong>
                    <span>{rowDetail}</span>
                  </div>
                  <StatusBadge value={rowDecision} />
                </article>
              )
            })}
          </div>
        ) : (
          <EmptyState icon={<ClipboardList size={24} />} text="Batch results will appear here." />
        )}
        <JsonBlock data={batchResult} emptyText="Raw batch response will appear here." />
      </section>
    </section>
  )
}
