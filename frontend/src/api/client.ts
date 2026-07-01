import { isRecord } from '../utils/format'

export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api').replace(/\/$/, '')

export async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  const contentType = response.headers.get('content-type') ?? ''
  const payload: unknown = contentType.includes('application/json')
    ? await response.json()
    : await response.text()

  if (!response.ok) {
    const message = isRecord(payload)
      ? String(payload.detail ?? payload.message ?? response.statusText)
      : String(payload || response.statusText)
    throw new Error(message)
  }

  return payload as T
}
