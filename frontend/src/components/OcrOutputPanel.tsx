import { Braces } from 'lucide-react'
import { buildStructuredOcrJson, compactOcrDebugPayload, hasData, isRecord } from '../utils/format'
import { EmptyState, JsonBlock, KeyValueGrid } from './common'

export function OcrOutputPanel({ result }: { result: unknown }) {
  if (!result) {
    return <EmptyState icon={<Braces size={22} />} text="AI structured JSON will appear here." />
  }
  if (!isRecord(result)) {
    return <JsonBlock data={result} />
  }

  const structured = buildStructuredOcrJson(result)
  const rawText = typeof result.raw_text === 'string' ? result.raw_text : ''
  const debugPayload = compactOcrDebugPayload(result)

  return (
    <div className="tool-output">
      <section className="result-section">
        <h3>AI Structured JSON</h3>
        <KeyValueGrid data={structured.document} />
        <JsonBlock data={structured.extracted_json} emptyText="No structured fields returned." />
      </section>

      {hasData(structured.review) && (
        <section className="result-section">
          <h3>AI Review</h3>
          <JsonBlock data={structured.review} />
        </section>
      )}

      {rawText && (
        <details className="details-block">
          <summary>Extracted Text</summary>
          <pre className="text-block">{rawText}</pre>
        </details>
      )}

      <details className="details-block">
        <summary>Raw OCR Debug Response</summary>
        <JsonBlock data={debugPayload} />
      </details>
    </div>
  )
}
