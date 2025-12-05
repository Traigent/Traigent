import { useEffect, useMemo, useRef, useState } from 'react'
import { copyIssueToClipboard } from '../utils/clipboard'
import type { IssueRecord } from '../types'

export type IssueColumnKey =
  | 'modulePath'
  | 'category'
  | 'id'
  | 'title'
  | 'severity'
  | 'location'
  | 'description'
  | 'impact'
  | 'scope'
  | 'priorityScore'
  | 'effort'
  | 'status'
  | 'lineSpan'
  | 'provenance'

export interface IssueColumn {
  key: IssueColumnKey
  label: string
  defaultVisible?: boolean
}

interface IssueTableProps {
  issues: IssueRecord[]
  columns: IssueColumn[]
  emptyMessage?: string
  enableColumnPicker?: boolean
  columnPickerHint?: string
  refreshKey?: number
  defaultSort?: {
    key: IssueColumnKey
    direction?: SortDirection
  }
}

type SortDirection = 'asc' | 'desc'

interface SortState {
  key: IssueColumnKey
  direction: SortDirection
}

const STATUS_FILTER_PATTERN = /(?:resolved|fixed)/i

const SEVERITY_WEIGHTS: Record<string, number> = {
  critical: 4,
  high: 3,
  medium: 2,
  low: 1,
}

const cellValue = (issue: IssueRecord, key: IssueColumnKey): string => {
  switch (key) {
    case 'scope':
      return issue.scope.length ? issue.scope.join(', ') : ''
    default:
      return (issue[key] ?? '') as string
  }
}

const displayValue = (value: string): string => (value.trim().length ? value : '—')

const cloneSet = (set?: Set<string>): Set<string> =>
  set ? new Set<string>(set) : new Set<string>()

const getSeverityWeight = (value: string): number =>
  SEVERITY_WEIGHTS[value.toLowerCase()] ?? 0

export const IssueTable = ({
  issues,
  columns,
  emptyMessage = 'No issues found',
  enableColumnPicker = false,
  columnPickerHint,
  refreshKey = 0,
  defaultSort,
}: IssueTableProps) => {
  const defaultVisibleColumns = useMemo(() => {
    const defaults = columns
      .filter((column) => column.defaultVisible !== false)
      .map((column) => column.key)
    return defaults.length > 0 ? defaults : columns.map((column) => column.key)
  }, [columns])

  const defaultSortState = useMemo<SortState | null>(
    () =>
      defaultSort
        ? { key: defaultSort.key, direction: defaultSort.direction ?? 'desc' }
        : null,
    [defaultSort?.key, defaultSort?.direction],
  )

  const buildInitialTextFilters = () =>
    columns.reduce<Record<IssueColumnKey, string>>((acc, column) => {
      acc[column.key] = ''
      return acc
    }, {} as Record<IssueColumnKey, string>)

  const buildInitialMultiFilters = () =>
    columns.reduce<Record<IssueColumnKey, Set<string>>>((acc, column) => {
      if (column.key === 'status') {
        const include = new Set<string>()
        for (const issue of issues) {
          const value = cellValue(issue, column.key)
          if (!STATUS_FILTER_PATTERN.test(value)) {
            include.add(value)
          }
        }
        acc[column.key] = include
      } else {
        acc[column.key] = new Set<string>()
      }
      return acc
    }, {} as Record<IssueColumnKey, Set<string>>)

  const [visibleColumns, setVisibleColumns] = useState<IssueColumnKey[]>(() => [
    ...defaultVisibleColumns,
  ])
  const [filters, setFilters] = useState<Record<IssueColumnKey, string>>(
    () => buildInitialTextFilters(),
  )
  const [multiFilters, setMultiFilters] = useState<Record<IssueColumnKey, Set<string>>>(
    () => buildInitialMultiFilters(),
  )
  const [sortState, setSortState] = useState<SortState | null>(defaultSortState)
  const [pickerOpen, setPickerOpen] = useState(false)
  const [selectedIssueId, setSelectedIssueId] = useState<string | null>(null)
  const [copyMessage, setCopyMessage] = useState<string | null>(null)
  const copyTimeoutRef = useRef<number | null>(null)
  const tableRef = useRef<HTMLDivElement | null>(null)
  const [activeFilterMenu, setActiveFilterMenu] = useState<IssueColumnKey | null>(null)
  const userModifiedFiltersRef = useRef<Set<IssueColumnKey>>(new Set())

  useEffect(
    () => () => {
      if (copyTimeoutRef.current !== null && typeof window !== 'undefined') {
        window.clearTimeout(copyTimeoutRef.current)
      }
    },
    [],
  )

  useEffect(() => {
    if (!defaultSortState) return
    setSortState((prev) => {
      if (
        !prev ||
        prev.key !== defaultSortState.key ||
        prev.direction !== defaultSortState.direction
      ) {
        return defaultSortState
      }
      return prev
    })
  }, [defaultSortState])

  useEffect(() => {
    setActiveFilterMenu(null)
  }, [refreshKey])

  useEffect(() => {
    if (!columns.some((column) => column.key === 'status')) return
    if (userModifiedFiltersRef.current.has('status')) return

    setMultiFilters((prev) => {
      const current = cloneSet(prev.status)
      let changed = false
      for (const issue of issues) {
        const value = cellValue(issue, 'status')
        if (value && !STATUS_FILTER_PATTERN.test(value) && !current.has(value)) {
          current.add(value)
          changed = true
        }
      }
      if (!prev.status || changed) {
        return { ...prev, status: current }
      }
      return prev
    })
  }, [issues, columns])

  useEffect(() => {
    if (typeof document === 'undefined') return undefined
    const handleClick = (event: MouseEvent) => {
      if (!tableRef.current || !(event.target instanceof Node)) return
      if (!tableRef.current.contains(event.target)) {
        setActiveFilterMenu(null)
      }
    }
    document.addEventListener('click', handleClick)
    return () => document.removeEventListener('click', handleClick)
  }, [])

  const handleSort = (key: IssueColumnKey) => {
    setSortState((prev) => {
      if (!prev || prev.key !== key) {
        return { key, direction: 'asc' }
      }
      if (prev.direction === 'asc') {
        return { key, direction: 'desc' }
      }
      return null
    })
  }

  const toggleColumnVisibility = (key: IssueColumnKey) => {
    setVisibleColumns((prev) => {
      const isVisible = prev.includes(key)
      if (isVisible) {
        return prev.filter((columnKey) => columnKey !== key)
      }
      return [...prev, key]
    })
    setFilters((prev) => ({ ...prev, [key]: '' }))
    setSortState((prev) => (prev?.key === key ? null : prev))
    setActiveFilterMenu((current) => (current === key ? null : current))
  }

  const activeColumns = useMemo(
    () => columns.filter((column) => visibleColumns.includes(column.key)),
    [columns, visibleColumns],
  )

  const filteredIssues = useMemo(
    () =>
      issues.filter((issue) =>
        activeColumns.every(({ key }) => {
          const filter = filters[key]
          const multiFilter = multiFilters[key]
          const value = cellValue(issue, key)
          const matchesText = filter ? value.toLowerCase().includes(filter.toLowerCase()) : true
          const matchesMulti =
            multiFilter && multiFilter.size > 0 ? multiFilter.has(value) : true
          return matchesText && matchesMulti
        }),
      ),
    [issues, activeColumns, filters, multiFilters],
  )

  const columnValueOptions = useMemo(
    () =>
      columns.reduce<Record<IssueColumnKey, string[]>>((acc, column) => {
        const values = new Set<string>()
        for (const issue of issues) {
          values.add(cellValue(issue, column.key))
        }
        acc[column.key] = Array.from(values).sort((a, b) =>
          a.localeCompare(b, undefined, { sensitivity: 'base' }),
        )
        return acc
      }, {} as Record<IssueColumnKey, string[]>),
    [issues, columns],
  )

  const sortedIssues = useMemo(() => {
    if (!sortState) return filteredIssues
    const { key, direction } = sortState
    if (!visibleColumns.includes(key)) return filteredIssues
    const sorted = [...filteredIssues]
    sorted.sort((a, b) => {
      const rawA = cellValue(a, key)
      const rawB = cellValue(b, key)
      if (key === 'severity') {
        const diff = getSeverityWeight(rawA) - getSeverityWeight(rawB)
        if (diff !== 0) {
          return direction === 'asc' ? diff : -diff
        }
      }
      const numA = Number(rawA)
      const numB = Number(rawB)
      if (!Number.isNaN(numA) && !Number.isNaN(numB)) {
        return direction === 'asc' ? numA - numB : numB - numA
      }
      const aValue = rawA.toLowerCase()
      const bValue = rawB.toLowerCase()
      if (aValue < bValue) return direction === 'asc' ? -1 : 1
      if (aValue > bValue) return direction === 'asc' ? 1 : -1
      return 0
    })
    return sorted
  }, [filteredIssues, sortState, visibleColumns])

  if (!issues.length) {
    return <p className="empty-message">{emptyMessage}</p>
  }

  return (
    <div className="table-container" ref={tableRef}>
      {enableColumnPicker && (
        <div className="column-controls">
          <button
            type="button"
            className="column-toggle"
            onClick={() => setPickerOpen((open) => !open)}
            aria-expanded={pickerOpen}
          >
            ⚙ Columns
          </button>
          {columnPickerHint && (
            <span className="hint-icon" title={columnPickerHint} aria-label="Column picker help">
              ⓘ
            </span>
          )}
        </div>
      )}
      {enableColumnPicker && pickerOpen && (
        <div className="column-menu">
          {columns.map((column) => {
            const isVisible = visibleColumns.includes(column.key)
            const disableToggle = isVisible && visibleColumns.length === 1
            return (
              <label key={column.key} className={disableToggle ? 'disabled' : ''}>
                <input
                  type="checkbox"
                  checked={isVisible}
                  disabled={disableToggle}
                  onChange={() => toggleColumnVisibility(column.key)}
                />
                <span>{column.label}</span>
              </label>
            )
          })}
        </div>
      )}
      {copyMessage && <div className="copy-toast">{copyMessage}</div>}
      <table className="issue-table">
        <thead>
          <tr>
            {activeColumns.map(({ key, label }) => (
              <th key={key}>
                <div className="th-content">
                  <button
                    type="button"
                    onClick={() => handleSort(key)}
                    className="sort-button"
                    aria-label={`Sort by ${label}`}
                  >
                    <span>{label}</span>
                    {sortState?.key === key && (
                      <span className="sort-indicator">
                        {sortState.direction === 'asc' ? '↑' : '↓'}
                      </span>
                    )}
                  </button>
                  <div className="filter-controls">
                    <input
                      type="text"
                      value={filters[key]}
                      onChange={(event) =>
                        setFilters((prev) => ({ ...prev, [key]: event.target.value }))
                      }
                      placeholder="Filter…"
                    />
                    <div className="filter-buttons">
                      <button
                        type="button"
                        className={`filter-menu-button ${
                          activeFilterMenu === key ? 'active' : ''
                        }`}
                        onClick={(event) => {
                          event.stopPropagation()
                          setActiveFilterMenu((current) => (current === key ? null : key))
                        }}
                        title="Select values"
                        aria-label={`Filter ${label} by value`}
                      >
                        ☰
                      </button>
                      <button
                        type="button"
                        className="filter-clear-button"
                        onClick={(event) => {
                          event.stopPropagation()
                          userModifiedFiltersRef.current.add(key)
                          setMultiFilters((prev) => ({ ...prev, [key]: new Set<string>() }))
                        }}
                        title="Clear selected values"
                        aria-label={`Clear selected ${label} filters`}
                      >
                        ✕
                      </button>
                    </div>
                    {activeFilterMenu === key && (
                      <div
                        className="value-menu"
                        onClick={(event) => event.stopPropagation()}
                      >
                        <div className="value-menu-header">
                          <span>Select values</span>
                          <button
                            type="button"
                            onClick={() => {
                              userModifiedFiltersRef.current.add(key)
                              setMultiFilters((prev) => ({ ...prev, [key]: new Set<string>() }))
                            }}
                          >
                            Clear all
                          </button>
                        </div>
                        <div className="value-menu-options">
                          {columnValueOptions[key]?.map((value) => {
                            const optionKey = value || '__EMPTY__'
                            const optionLabel = value || '—'
                            const isChecked = multiFilters[key]?.has(value) ?? false
                            return (
                              <label key={optionKey}>
                                <input
                                  type="checkbox"
                                  checked={isChecked}
                                  onChange={() => {
                                    userModifiedFiltersRef.current.add(key)
                                    setMultiFilters((prev) => {
                                      const next = { ...prev }
                                      const nextSet = cloneSet(prev[key])
                                      if (nextSet.has(value)) {
                                        nextSet.delete(value)
                                      } else {
                                        nextSet.add(value)
                                      }
                                      next[key] = nextSet
                                      return next
                                    })
                                  }}
                                />
                                <span>{optionLabel}</span>
                              </label>
                            )
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedIssues.length === 0 ? (
            <tr>
              <td colSpan={activeColumns.length || 1} className="empty-row">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            sortedIssues.map((issue) => (
              <tr
                key={`${issue.sourcePath}-${issue.id}-${issue.title}`}
                className={
                  selectedIssueId === `${issue.sourcePath}-${issue.id}-${issue.title}`
                    ? 'selected-row'
                    : ''
                }
                onClick={async () => {
                  const issueKey = `${issue.sourcePath}-${issue.id}-${issue.title}`
                  setSelectedIssueId(issueKey)
                  try {
                    await copyIssueToClipboard(issue)
                    setCopyMessage('Copied issue JSON to clipboard')
                  } catch (error) {
                    console.error('Failed to copy issue', error)
                    setCopyMessage('Unable to copy issue - see console for details')
                  }
                  if (copyTimeoutRef.current !== null && typeof window !== 'undefined') {
                    window.clearTimeout(copyTimeoutRef.current)
                  }
                  if (typeof window !== 'undefined') {
                    copyTimeoutRef.current = window.setTimeout(() => {
                      setCopyMessage(null)
                      copyTimeoutRef.current = null
                    }, 2000)
                  }
                }}
              >
                {activeColumns.map(({ key }) => (
                  <td key={key}>{displayValue(cellValue(issue, key))}</td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}

