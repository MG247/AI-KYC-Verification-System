export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function parseJsonObject(value: string, label: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(value)
    if (!isRecord(parsed)) {
      throw new Error(`${label} must be a JSON object.`)
    }
    return parsed
  } catch (error) {
    if (error instanceof SyntaxError) {
      throw new Error(`${label} contains invalid JSON.`, { cause: error })
    }
    throw error
  }
}

export function formatLabel(value: string) {
  return value
    .replace(/_/g, ' ')
    .replace(/\w\S*/g, (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === '') {
    return 'Not returned'
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? String(value) : value.toFixed(3)
  }
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No'
  }
  if (typeof value === 'object') {
    return JSON.stringify(value, null, 2)
  }
  return String(value)
}

export function scorePercent(value: unknown) {
  const numericValue = typeof value === 'number' ? value : Number(value)
  if (Number.isNaN(numericValue)) {
    return 0
  }
  return Math.max(0, Math.min(100, numericValue * 100))
}

export function decisionTone(value?: string) {
  const normalized = (value ?? '').toUpperCase()
  if (normalized.includes('APPROVE') || normalized.includes('AUTHENTIC') || normalized.includes('PASS')) {
    return 'success'
  }
  if (normalized.includes('REJECT') || normalized.includes('TAMPER') || normalized.includes('FAKE')) {
    return 'danger'
  }
  if (normalized.includes('REVIEW') || normalized.includes('MISMATCH') || normalized.includes('LOW')) {
    return 'warning'
  }
  return 'neutral'
}

export function buildStructuredOcrJson(result: Record<string, unknown>) {
  const generic = isRecord(result.generic_analysis) ? result.generic_analysis : {}
  const dynamicFields = firstRecord(result.dynamic_fields, generic.dynamic_fields)
  const typedFields = cleanFieldRecord(result.typed_fields)
  const legacyFields = cleanFieldRecord(result.fields)
  const rawText = typeof result.raw_text === 'string' ? result.raw_text : ''
  const extractedJson = firstRecord(
    dynamicFields,
    typedFields,
    legacyFields,
    extractKeyValueJson(rawText),
  )
  const documentLabel = String(
    result.document_type_label
      ?? generic.document_type_label
      ?? result.document_type
      ?? 'Unknown Document',
  )

  return {
    document: {
      detected_label: documentLabel,
      detected_type: result.document_type ?? 'UNKNOWN',
      category: generic.document_category ?? 'OTHER',
      ocr_engine: result.ocr_engine ?? 'paddle',
      classification_confidence: result.classification_confidence ?? 0,
      ocr_confidence: result.confidence_score ?? 0,
    },
    extracted_json: extractedJson,
    review: {
      summary: generic.summary,
      visual_concerns: generic.visual_concerns,
      extraction_notes: generic.extraction_notes,
      source: generic.source,
    },
  }
}

export function compactOcrDebugPayload(result: Record<string, unknown>) {
  const { lines, words, raw_text, ...rest } = result
  return {
    ...rest,
    raw_text_preview: typeof raw_text === 'string' ? raw_text.slice(0, 1500) : raw_text,
    line_count: Array.isArray(lines) ? lines.length : 0,
    word_count: Array.isArray(words) ? words.length : 0,
  }
}

function cleanFieldRecord(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) return {}
  const ignored = new Set(['schema_type', 'qr_codes', 'tables'])
  return Object.fromEntries(
    Object.entries(value).filter(([key, item]) => !ignored.has(key) && isMeaningfulValue(item)),
  )
}

function firstRecord(...records: unknown[]): Record<string, unknown> {
  for (const record of records) {
    if (isRecord(record) && hasData(record)) return record
  }
  return {}
}

function extractKeyValueJson(rawText: string): Record<string, unknown> {
  const fields: Record<string, unknown> = {}
  for (const line of rawText.split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z][A-Za-z0-9\s/&().,#-]{1,52})\s*[:-]\s*(.{2,220})\s*$/)
    if (!match) continue
    const key = match[1].trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
    const value = match[2].trim()
    if (!key || !value) continue
    if (fields[key]) {
      fields[key] = Array.isArray(fields[key]) ? [...fields[key], value] : [fields[key], value]
    } else {
      fields[key] = value
    }
  }
  return fields
}

export function hasData(value: unknown) {
  if (Array.isArray(value)) return value.length > 0
  if (isRecord(value)) return Object.values(value).some(isMeaningfulValue)
  return isMeaningfulValue(value)
}

function isMeaningfulValue(value: unknown) {
  return value !== null && value !== undefined && value !== '' && !(Array.isArray(value) && value.length === 0)
    && !(isRecord(value) && Object.keys(value).length === 0)
}
