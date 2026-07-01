import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  BadgeCheck,
  ClipboardList,
  FileScan,
  Gauge,
  HeartPulse,
  History,
  Loader2,
  Moon,
  RefreshCcw,
  ShieldAlert,
  ShieldCheck,
  Sun,
  WandSparkles,
} from 'lucide-react'
import { requestJson } from './api/client'
import { AuditPage } from './pages/AuditPage'
import { BatchPage } from './pages/BatchPage'
import { ToolsPage } from './pages/ToolsPage'
import { VerifyPage } from './pages/VerifyPage'
import type { HealthResponse, KycResponse, MetricsResponse, TabKey } from './types'
import './App.css'

type Theme = 'light' | 'dark'

const THEME_STORAGE_KEY = 'kyc-ai-theme'

function getInitialTheme(): Theme {
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
  if (storedTheme === 'light' || storedTheme === 'dark') {
    return storedTheme
  }
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

const tabs = [
  { key: 'verify' as const, label: 'Verify', Icon: FileScan },
  { key: 'tools' as const, label: 'Tools', Icon: WandSparkles },
  { key: 'batch' as const, label: 'Batch', Icon: ClipboardList },
  { key: 'audit' as const, label: 'Audit', Icon: History },
]

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('verify')
  const [theme, setTheme] = useState<Theme>(getInitialTheme)
  const [isStatusLoading, setIsStatusLoading] = useState(false)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [latestRequestId, setLatestRequestId] = useState('')

  const refreshStatus = useCallback(async () => {
    setIsStatusLoading(true)
    setStatusError(null)
    const [healthResponse, metricsResponse] = await Promise.allSettled([
      requestJson<HealthResponse>('/healthz'),
      requestJson<MetricsResponse>('/metrics'),
    ])

    if (healthResponse.status === 'fulfilled') {
      setHealth(healthResponse.value)
    } else {
      setHealth(null)
      setStatusError(healthResponse.reason instanceof Error ? healthResponse.reason.message : 'API offline')
    }

    if (metricsResponse.status === 'fulfilled') {
      setMetrics(metricsResponse.value)
    } else {
      setMetrics(null)
    }

    setIsStatusLoading(false)
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem(THEME_STORAGE_KEY, theme)
  }, [theme])

  useEffect(() => {
    const statusTimer = window.setTimeout(() => {
      void refreshStatus()
    }, 0)

    return () => window.clearTimeout(statusTimer)
  }, [refreshStatus])

  const statusTone = statusError ? 'danger' : health?.status === 'ok' ? 'success' : 'warning'

  const metricCards = useMemo(
    () => [
      {
        label: 'Requests',
        value: metrics?.requests_total ?? 0,
        detail: `${metrics?.errors_total ?? 0} errors`,
        Icon: Activity,
      },
      {
        label: 'Approved',
        value: metrics?.decisions?.APPROVE ?? 0,
        detail: 'Final decisions',
        Icon: ShieldCheck,
      },
      {
        label: 'Review',
        value: metrics?.decisions?.REVIEW ?? 0,
        detail: 'Manual queue',
        Icon: ShieldAlert,
      },
      {
        label: 'P95 Latency',
        value: `${metrics?.latency_ms?.p95 ?? 0} ms`,
        detail: `${metrics?.latency_ms?.samples ?? 0} samples`,
        Icon: Gauge,
      },
    ],
    [metrics],
  )

  function handleVerified(result: KycResponse) {
    setLatestRequestId(result.request_id)
  }

  function toggleTheme() {
    setTheme((currentTheme) => (currentTheme === 'dark' ? 'light' : 'dark'))
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true">
            <BadgeCheck size={22} />
          </span>
          <div>
            <h1>KYC AI Console</h1>
            <p>Document OCR, validation, forensic checks, risk scoring, and audit review.</p>
          </div>
        </div>

        <div className="status-cluster">
          <span className={`status-pill ${statusTone}`}>
            <HeartPulse size={16} />
            {statusError ? 'API offline' : health?.status === 'ok' ? 'API ready' : 'Checking API'}
          </span>
          <button
            type="button"
            className="icon-button"
            onClick={toggleTheme}
            title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
            aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          >
            {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={() => void refreshStatus()}
            title="Refresh API status"
            aria-label="Refresh API status"
          >
            {isStatusLoading ? <Loader2 className="spin" size={17} /> : <RefreshCcw size={17} />}
          </button>
        </div>
      </header>

      <nav className="tabbar" aria-label="KYC workflows">
        {tabs.map(({ key, label, Icon }) => (
          <button
            key={key}
            type="button"
            className={activeTab === key ? 'active' : ''}
            onClick={() => setActiveTab(key)}
          >
            <Icon size={18} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <main className="workspace">
        <section className="status-grid" aria-label="Pipeline metrics">
          {metricCards.map(({ label, value, detail, Icon }) => (
            <article className="metric-card" key={label}>
              <div>
                <span>{label}</span>
                <strong>{value}</strong>
                <small>{detail}</small>
              </div>
              <Icon size={22} />
            </article>
          ))}
        </section>

        {statusError && (
          <div className="banner danger" role="status">
            <AlertTriangle size={17} />
            <span>{statusError}</span>
          </div>
        )}

        {activeTab === 'verify' && (
          <VerifyPage onVerified={handleVerified} onStatusRefresh={refreshStatus} />
        )}
        {activeTab === 'tools' && <ToolsPage />}
        {activeTab === 'batch' && (
          <BatchPage onRequestIdFound={setLatestRequestId} onStatusRefresh={refreshStatus} />
        )}
        {activeTab === 'audit' && <AuditPage latestRequestId={latestRequestId} />}
      </main>
    </div>
  )
}

export default App
