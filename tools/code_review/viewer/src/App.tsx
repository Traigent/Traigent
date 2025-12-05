import { useEffect, useMemo, useState } from 'react'
import { IssueTable } from './components/IssueTable'
import type { IssueColumn } from './components/IssueTable'
import { TabBar } from './components/TabBar'
import { loadReviewData, loadValidatedData } from './dataLoader'
import type { ReviewCategory } from './types'
import './App.css'

const reviewData = loadReviewData()
const validatedData = loadValidatedData()

const CATEGORY_LABELS: Record<ReviewCategory, string> = {
  performance: 'Performance',
  code_quality: 'Code Quality',
  soundness_correctness: 'Soundness & Correctness',
}

const CATEGORY_ORDER: ReviewCategory[] = [
  'soundness_correctness',
  'performance',
  'code_quality',
]

const ISSUE_TYPE_COLUMNS: IssueColumn[] = [
  { key: 'modulePath', label: 'Module' },
  { key: 'id', label: 'Issue ID' },
  { key: 'title', label: 'Title' },
  { key: 'severity', label: 'Severity' },
  { key: 'location', label: 'Location' },
  { key: 'description', label: 'Description' },
  { key: 'impact', label: 'Impact' },
  { key: 'scope', label: 'Scope' },
]

const MODULE_COLUMNS: IssueColumn[] = [
  { key: 'category', label: 'Category' },
  { key: 'id', label: 'Issue ID' },
  { key: 'title', label: 'Title' },
  { key: 'severity', label: 'Severity' },
  { key: 'location', label: 'Location' },
  { key: 'description', label: 'Description' },
  { key: 'impact', label: 'Impact' },
  { key: 'scope', label: 'Scope' },
]

const FOLDER_COLUMNS: IssueColumn[] = [
  { key: 'modulePath', label: 'Module' },
  { key: 'category', label: 'Category' },
  { key: 'id', label: 'Issue ID' },
  { key: 'title', label: 'Title' },
  { key: 'severity', label: 'Severity' },
  { key: 'location', label: 'Location' },
  { key: 'description', label: 'Description' },
  { key: 'impact', label: 'Impact' },
  { key: 'scope', label: 'Scope' },
]

const VALIDATED_COLUMNS: IssueColumn[] = [
  { key: 'modulePath', label: 'Module' },
  { key: 'category', label: 'Category' },
  { key: 'id', label: 'Issue ID' },
  { key: 'title', label: 'Title' },
  { key: 'severity', label: 'Severity' },
  { key: 'priorityScore', label: 'Priority Score' },
  { key: 'status', label: 'Status' },
  { key: 'lineSpan', label: 'Line Span' },
  { key: 'effort', label: 'Effort', defaultVisible: false },
  { key: 'location', label: 'Location', defaultVisible: false },
  { key: 'provenance', label: 'Provenance', defaultVisible: false },
  { key: 'description', label: 'Description', defaultVisible: false },
  { key: 'impact', label: 'Impact', defaultVisible: false },
  { key: 'scope', label: 'Scope', defaultVisible: false },
]

type PrimaryTab = 'raw' | 'validated'
type ViewTab = 'issue-types' | 'modules' | 'folders'
type SortDirection = 'asc' | 'desc'

const coverageClassName = (coverage: number): string => {
  if (coverage >= 1) return 'coverage-badge coverage-complete'
  if (coverage >= 0.8) return 'coverage-badge coverage-warning'
  return 'coverage-badge coverage-risk'
}

const formatPercent = (value: number): string => `${Math.round(value * 100)}%`

function App() {
  const [primaryTab, setPrimaryTab] = useState<PrimaryTab>('raw')
  const [rawSubTab, setRawSubTab] = useState<ViewTab>('issue-types')
  const defaultCategory =
    CATEGORY_ORDER.find(
      (category) => reviewData.issuesByCategory[category]?.length > 0,
    ) ?? CATEGORY_ORDER[0]
  const [activeCategory, setActiveCategory] = useState<ReviewCategory>(defaultCategory)

  const defaultValidatedCategory =
    CATEGORY_ORDER.find(
      (category) => validatedData.issuesByCategory[category]?.length > 0,
    ) ?? CATEGORY_ORDER[0]
  const [validatedCategory, setValidatedCategory] = useState<ReviewCategory>(
    defaultValidatedCategory,
  )

  const firstModule = reviewData.modules[0]?.modulePath ?? ''
  const [selectedModule, setSelectedModule] = useState(firstModule)
  const firstFolder = reviewData.folders[0]?.folderPath ?? ''
  const [selectedFolder, setSelectedFolder] = useState(firstFolder)

  const firstValidatedModule = validatedData.modules[0]?.modulePath ?? ''
  const [validatedModule, setValidatedModule] = useState(firstValidatedModule)
  const firstValidatedFolder = validatedData.folders[0]?.folderPath ?? ''
  const [validatedFolder, setValidatedFolder] = useState(firstValidatedFolder)

  const [moduleCategorySort, setModuleCategorySort] = useState<{
    key: 'label' | 'count'
    direction: SortDirection
  } | null>(null)
  const [folderModuleSort, setFolderModuleSort] = useState<{
    key: 'modulePath' | 'issueCount' | 'reviewCount'
    direction: SortDirection
  } | null>(null)
  const [validatedModuleCategorySort, setValidatedModuleCategorySort] = useState<{
    key: 'label' | 'count'
    direction: SortDirection
  } | null>(null)
  const [validatedFolderModuleSort, setValidatedFolderModuleSort] = useState<{
    key: 'modulePath' | 'issueCount' | 'reviewCount'
    direction: SortDirection
  } | null>(null)
  const [validatedSubTab, setValidatedSubTab] = useState<ViewTab>('issue-types')
  const [refreshCounter, setRefreshCounter] = useState(0)

  const categoryTabs = useMemo(
    () =>
      CATEGORY_ORDER.map((category) => ({
        id: category,
        label: CATEGORY_LABELS[category],
        badge: reviewData.issuesByCategory[category]?.length ?? 0,
      })),
    [],
  )

  const validatedCategoryTabs = useMemo(
    () =>
      CATEGORY_ORDER.map((category) => ({
        id: category,
        label: CATEGORY_LABELS[category],
        badge: validatedData.issuesByCategory[category]?.length ?? 0,
      })),
    [],
  )

  const selectedModuleData = reviewData.modules.find(
    (module) => module.modulePath === selectedModule,
  )
  const selectedFolderData = reviewData.folders.find(
    (folder) => folder.folderPath === selectedFolder,
  )

  const validatedSelectedModuleData = validatedData.modules.find(
    (module) => module.modulePath === validatedModule,
  )
  const validatedSelectedFolderData = validatedData.folders.find(
    (folder) => folder.folderPath === validatedFolder,
  )

  const categorizedIssues = reviewData.issuesByCategory[activeCategory] ?? []
  const validatedCategorizedIssues =
    validatedData.issuesByCategory[validatedCategory] ?? []

  useEffect(() => {
    setModuleCategorySort(null)
  }, [selectedModule])

  useEffect(() => {
    setFolderModuleSort(null)
  }, [selectedFolder])

  useEffect(() => {
    setValidatedModuleCategorySort(null)
  }, [validatedModule])

  useEffect(() => {
    setValidatedFolderModuleSort(null)
  }, [validatedFolder])

  const toggleModuleCategorySort = (key: 'label' | 'count') => {
    setModuleCategorySort((prev) => {
      if (!prev || prev.key !== key) {
        return { key, direction: 'asc' }
      }
      if (prev.direction === 'asc') return { key, direction: 'desc' }
      return null
    })
  }

  const toggleFolderModuleSort = (key: 'modulePath' | 'issueCount' | 'reviewCount') => {
    setFolderModuleSort((prev) => {
      if (!prev || prev.key !== key) {
        return { key, direction: 'asc' }
      }
      if (prev.direction === 'asc') return { key, direction: 'desc' }
      return null
    })
  }

  const toggleValidatedModuleCategorySort = (key: 'label' | 'count') => {
    setValidatedModuleCategorySort((prev) => {
      if (!prev || prev.key !== key) {
        return { key, direction: 'asc' }
      }
      if (prev.direction === 'asc') return { key, direction: 'desc' }
      return null
    })
  }

  const toggleValidatedFolderModuleSort = (
    key: 'modulePath' | 'issueCount' | 'reviewCount',
  ) => {
    setValidatedFolderModuleSort((prev) => {
      if (!prev || prev.key !== key) {
        return { key, direction: 'asc' }
      }
      if (prev.direction === 'asc') return { key, direction: 'desc' }
      return null
    })
  }

  const moduleCategoryRows = useMemo(() => {
    if (!selectedModuleData) return []
    const rows = (Object.entries(selectedModuleData.categoryCounts) as [
      ReviewCategory,
      number,
    ][]).map(([category, count]) => ({
      category,
      label: CATEGORY_LABELS[category],
      count,
    }))
    if (!moduleCategorySort) return rows
    const { key, direction } = moduleCategorySort
    return [...rows].sort((a, b) => {
      const aValue = a[key]
      const bValue = b[key]
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return direction === 'asc' ? aValue - bValue : bValue - aValue
      }
      const aString = String(aValue).toLowerCase()
      const bString = String(bValue).toLowerCase()
      if (aString < bString) return direction === 'asc' ? -1 : 1
      if (aString > bString) return direction === 'asc' ? 1 : -1
      return 0
    })
  }, [selectedModuleData, moduleCategorySort])

  const validatedModuleCategoryRows = useMemo(() => {
    if (!validatedSelectedModuleData) return []
    const rows = (Object.entries(validatedSelectedModuleData.categoryCounts) as [
      ReviewCategory,
      number,
    ][]).map(([category, count]) => ({
      category,
      label: CATEGORY_LABELS[category],
      count,
    }))
    if (!validatedModuleCategorySort) return rows
    const { key, direction } = validatedModuleCategorySort
    return [...rows].sort((a, b) => {
      const aValue = a[key]
      const bValue = b[key]
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return direction === 'asc' ? aValue - bValue : bValue - aValue
      }
      const aString = String(aValue).toLowerCase()
      const bString = String(bValue).toLowerCase()
      if (aString < bString) return direction === 'asc' ? -1 : 1
      if (aString > bString) return direction === 'asc' ? 1 : -1
      return 0
    })
  }, [validatedSelectedModuleData, validatedModuleCategorySort])

  const folderModuleRows = useMemo(() => {
    if (!selectedFolderData) return []
    const rows = selectedFolderData.moduleStats
    if (!folderModuleSort) return rows
    const { key, direction } = folderModuleSort
    return [...rows].sort((a, b) => {
      const aValue = a[key]
      const bValue = b[key]
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return direction === 'asc' ? aValue - bValue : bValue - aValue
      }
      const aString = String(aValue).toLowerCase()
      const bString = String(bValue).toLowerCase()
      if (aString < bString) return direction === 'asc' ? -1 : 1
      if (aString > bString) return direction === 'asc' ? 1 : -1
      return 0
    })
  }, [selectedFolderData, folderModuleSort])

  const validatedFolderModuleRows = useMemo(() => {
    if (!validatedSelectedFolderData) return []
    const rows = validatedSelectedFolderData.moduleStats
    if (!validatedFolderModuleSort) return rows
    const { key, direction } = validatedFolderModuleSort
    return [...rows].sort((a, b) => {
      const aValue = a[key]
      const bValue = b[key]
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return direction === 'asc' ? aValue - bValue : bValue - aValue
      }
      const aString = String(aValue).toLowerCase()
      const bString = String(bValue).toLowerCase()
      if (aString < bString) return direction === 'asc' ? -1 : 1
      if (aString > bString) return direction === 'asc' ? 1 : -1
      return 0
    })
  }, [validatedSelectedFolderData, validatedFolderModuleSort])

  useEffect(() => {
    if (validatedData.modules.length === 0) {
      setValidatedModule('')
      return
    }
    setValidatedModule((current) => {
      if (!current) return validatedData.modules[0].modulePath
      return validatedData.modules.some((module) => module.modulePath === current)
        ? current
        : validatedData.modules[0].modulePath
    })
  }, [])

  useEffect(() => {
    if (validatedData.folders.length === 0) {
      setValidatedFolder('')
      return
    }
    setValidatedFolder((current) => {
      if (!current) return validatedData.folders[0].folderPath
      return validatedData.folders.some((folder) => folder.folderPath === current)
        ? current
        : validatedData.folders[0].folderPath
    })
  }, [])

  return (
    <div className="app-shell">
      <header>
        <h1>Review Issue Browser</h1>
        <p className="subtitle">
          Explore generated review reports by track, module, and folder. Use the filters on each
          column to refine results.
        </p>
        <button
          type="button"
          className="refresh-button"
          onClick={() => setRefreshCounter((count) => count + 1)}
          title="Refresh tables"
          aria-label="Refresh tables"
        >
          ↻
        </button>
      </header>

      <TabBar
        tabs={[
          { id: 'raw', label: 'Issues (raw)' },
          { id: 'validated', label: 'Validated Issues' },
        ]}
        activeId={primaryTab}
        onSelect={(id) => setPrimaryTab(id as PrimaryTab)}
      />

      {primaryTab === 'raw' && (
        <>
          <TabBar
            tabs={[
              { id: 'issue-types', label: 'Issues by Type' },
              { id: 'modules', label: 'Module View' },
              { id: 'folders', label: 'Folder View' },
            ]}
            activeId={rawSubTab}
            onSelect={(id) => setRawSubTab(id as ViewTab)}
          />
          {rawSubTab === 'issue-types' && (
            <section>
              {categoryTabs.length === 0 ? (
                <p className="empty-message">No review issues discovered.</p>
              ) : (
                <>
                  <h2 className="table-title">{CATEGORY_LABELS[activeCategory]} Issues</h2>
                  <TabBar
                    tabs={categoryTabs}
                    activeId={activeCategory}
                    onSelect={(id) => setActiveCategory(id as ReviewCategory)}
                  />
              <IssueTable
                issues={categorizedIssues}
                columns={ISSUE_TYPE_COLUMNS}
                emptyMessage="No issues recorded for this category."
                refreshKey={refreshCounter}
                defaultSort={{ key: 'severity', direction: 'desc' }}
              />
                </>
              )}
            </section>
          )}

          {rawSubTab === 'modules' && (
            <section>
              {reviewData.modules.length === 0 ? (
                <p className="empty-message">No modules with review data available.</p>
              ) : (
                <>
                  <div className={selectedModuleData ? 'module-header' : 'module-header disabled'}>
                    <label htmlFor="module-select">Module</label>
                    <select
                      id="module-select"
                      value={selectedModule}
                      onChange={(event) => setSelectedModule(event.target.value)}
                    >
                      {reviewData.modules.map((module) => (
                        <option key={module.modulePath} value={module.modulePath}>
                          {module.modulePath}
                        </option>
                      ))}
                    </select>
                  </div>

                  {selectedModuleData ? (
                    <>
                      <div className="stats-panel">
                        <h2>Issue Counts by Type</h2>
                        <table className="stats-table">
                          <thead>
                            <tr>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleModuleCategorySort('label')}
                                >
                                  <span>Category</span>
                                  {moduleCategorySort?.key === 'label' && (
                                    <span className="sort-indicator">
                                      {moduleCategorySort.direction === 'asc' ? '↑' : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleModuleCategorySort('count')}
                                >
                                  <span>Issues</span>
                                  {moduleCategorySort?.key === 'count' && (
                                    <span className="sort-indicator">
                                      {moduleCategorySort.direction === 'asc' ? '↑' : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {moduleCategoryRows.map((row) => (
                              <tr key={row.category}>
                                <td>{row.label}</td>
                                <td>{row.count}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="meta">
                          Review reports parsed for this module: {selectedModuleData.reviewCount}
                        </p>
                      </div>
                      <h2 className="table-title">All Issues for {selectedModule}</h2>
                  <IssueTable
                    issues={selectedModuleData.issues}
                    columns={MODULE_COLUMNS}
                    emptyMessage="This module has no recorded issues."
                    refreshKey={refreshCounter}
                    defaultSort={{ key: 'severity', direction: 'desc' }}
                  />
                    </>
                  ) : (
                    <p className="empty-message">Select a module to view its details.</p>
                  )}
                </>
              )}
            </section>
          )}

          {rawSubTab === 'folders' && (
            <section>
              {reviewData.folders.length === 0 ? (
                <p className="empty-message">No folders with aggregated review data.</p>
              ) : (
                <>
                  <div className="module-header">
                    <label htmlFor="folder-select">Folder</label>
                    <select
                      id="folder-select"
                      value={selectedFolder}
                      onChange={(event) => setSelectedFolder(event.target.value)}
                    >
                      {reviewData.folders.map((folder) => (
                        <option key={folder.folderPath} value={folder.folderPath}>
                          {folder.folderPath}
                        </option>
                      ))}
                    </select>
                  </div>

                  {selectedFolderData ? (
                    <>
                      <div className="stats-panel">
                        <h2>Module Statistics</h2>
                        <p className="meta">
                          Total review files across folder: {selectedFolderData.reviewCount}
                        </p>
                        <p className="meta">
                          Python modules: {selectedFolderData.totalModules} (reviewed:{' '}
                          {selectedFolderData.reviewedModules})
                        </p>
                        <div className={coverageClassName(selectedFolderData.coverage)}>
                          Coverage: {formatPercent(selectedFolderData.coverage)}
                        </div>
                        <table className="stats-table">
                          <thead>
                            <tr>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleFolderModuleSort('modulePath')}
                                >
                                  <span>Module</span>
                                  {folderModuleSort?.key === 'modulePath' && (
                                    <span className="sort-indicator">
                                      {folderModuleSort.direction === 'asc' ? '↑' : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleFolderModuleSort('issueCount')}
                                >
                                  <span>Issues</span>
                                  {folderModuleSort?.key === 'issueCount' && (
                                    <span className="sort-indicator">
                                      {folderModuleSort.direction === 'asc' ? '↑' : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleFolderModuleSort('reviewCount')}
                                >
                                  <span>Reviews</span>
                                  {folderModuleSort?.key === 'reviewCount' && (
                                    <span className="sort-indicator">
                                      {folderModuleSort.direction === 'asc' ? '↑' : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {folderModuleRows.map((stat) => (
                              <tr key={stat.modulePath}>
                                <td>{stat.modulePath}</td>
                                <td>{stat.issueCount}</td>
                                <td>{stat.reviewCount}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <h2 className="table-title">
                        Folder Issues ({CATEGORY_LABELS.performance}, {CATEGORY_LABELS.soundness_correctness}, {CATEGORY_LABELS.code_quality})
                      </h2>
                  <IssueTable
                    issues={selectedFolderData.issues}
                    columns={FOLDER_COLUMNS}
                    emptyMessage="This folder has no issues in scope."
                    refreshKey={refreshCounter}
                    defaultSort={{ key: 'severity', direction: 'desc' }}
                  />
                    </>
                  ) : (
                    <p className="empty-message">Select a folder to inspect aggregated results.</p>
                  )}
                </>
              )}
            </section>
          )}
        </>
      )}

      {primaryTab === 'validated' && (
        <section>
          <TabBar
            tabs={[
              { id: 'issue-types', label: 'Issues by Type' },
              { id: 'modules', label: 'Module View' },
              { id: 'folders', label: 'Folder View' },
            ]}
            activeId={validatedSubTab}
            onSelect={(id) => setValidatedSubTab(id as ViewTab)}
          />

          {validatedSubTab === 'issue-types' && (
            <div>
              {validatedCategoryTabs.every((tab) => tab.badge === 0) ? (
                <p className="empty-message">No validated issues available.</p>
              ) : (
                <>
                  <h2 className="table-title">
                    Validated {CATEGORY_LABELS[validatedCategory]} Issues
                  </h2>
                  <TabBar
                    tabs={validatedCategoryTabs}
                    activeId={validatedCategory}
                    onSelect={(id) => setValidatedCategory(id as ReviewCategory)}
                  />
                  <IssueTable
                    issues={validatedCategorizedIssues}
                    columns={VALIDATED_COLUMNS}
                    emptyMessage="No validated issues in this category."
                    enableColumnPicker
                    columnPickerHint="Toggle columns from the validated remediation report."
                    refreshKey={refreshCounter}
                    defaultSort={{ key: 'priorityScore', direction: 'desc' }}
                  />
                </>
              )}
            </div>
          )}

          {validatedSubTab === 'modules' && (
            <div>
              {validatedData.modules.length === 0 ? (
                <p className="empty-message">No validated module reports yet.</p>
              ) : (
                <>
                  <div
                    className={
                      validatedSelectedModuleData
                        ? 'module-header'
                        : 'module-header disabled'
                    }
                  >
                    <label htmlFor="validated-module-select">Module</label>
                    <select
                      id="validated-module-select"
                      value={validatedModule}
                      onChange={(event) => setValidatedModule(event.target.value)}
                    >
                      {validatedData.modules.map((module) => (
                        <option key={module.modulePath} value={module.modulePath}>
                          {module.modulePath}
                        </option>
                      ))}
                    </select>
                  </div>

                  {validatedSelectedModuleData ? (
                    <>
                      <div className="stats-panel">
                        <h2>Validated Issue Counts by Type</h2>
                        <table className="stats-table">
                          <thead>
                            <tr>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleValidatedModuleCategorySort('label')}
                                >
                                  <span>Category</span>
                                  {validatedModuleCategorySort?.key === 'label' && (
                                    <span className="sort-indicator">
                                      {validatedModuleCategorySort.direction === 'asc'
                                        ? '↑'
                                        : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleValidatedModuleCategorySort('count')}
                                >
                                  <span>Issues</span>
                                  {validatedModuleCategorySort?.key === 'count' && (
                                    <span className="sort-indicator">
                                      {validatedModuleCategorySort.direction === 'asc'
                                        ? '↑'
                                        : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {validatedModuleCategoryRows.map((row) => (
                              <tr key={row.category}>
                                <td>{row.label}</td>
                                <td>{row.count}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="meta">
                          Validated issues tracked for this module:{' '}
                          {validatedSelectedModuleData.reviewCount}
                        </p>
                      </div>
                      <h2 className="table-title">
                        Validated Issues for {validatedModule}
                      </h2>
                      <IssueTable
                        issues={validatedSelectedModuleData.issues}
                        columns={VALIDATED_COLUMNS}
                        emptyMessage="No validated issues recorded for this module."
                        enableColumnPicker
                        columnPickerHint="Select which validated report columns to display."
                        refreshKey={refreshCounter}
                        defaultSort={{ key: 'priorityScore', direction: 'desc' }}
                      />
                    </>
                  ) : (
                    <p className="empty-message">Select a module to inspect validated issues.</p>
                  )}
                </>
              )}
            </div>
          )}

          {validatedSubTab === 'folders' && (
            <div>
              {validatedData.folders.length === 0 ? (
                <p className="empty-message">No validated folder rollups yet.</p>
              ) : (
                <>
                  <div className="module-header">
                    <label htmlFor="validated-folder-select">Folder</label>
                    <select
                      id="validated-folder-select"
                      value={validatedFolder}
                      onChange={(event) => setValidatedFolder(event.target.value)}
                    >
                      {validatedData.folders.map((folder) => (
                        <option key={folder.folderPath} value={folder.folderPath}>
                          {folder.folderPath}
                        </option>
                      ))}
                    </select>
                  </div>

                  {validatedSelectedFolderData ? (
                    <>
                      <div className="stats-panel">
                        <h2>Validated Module Coverage</h2>
                        <p className="meta">
                          Total validated issues across folder: {validatedSelectedFolderData.reviewCount}
                        </p>
                        <p className="meta">
                          Python modules: {validatedSelectedFolderData.totalModules} (validated:{' '}
                          {validatedSelectedFolderData.reviewedModules})
                        </p>
                        <div className={coverageClassName(validatedSelectedFolderData.coverage)}>
                          Coverage: {formatPercent(validatedSelectedFolderData.coverage)}
                        </div>
                        <table className="stats-table">
                          <thead>
                            <tr>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleValidatedFolderModuleSort('modulePath')}
                                >
                                  <span>Module</span>
                                  {validatedFolderModuleSort?.key === 'modulePath' && (
                                    <span className="sort-indicator">
                                      {validatedFolderModuleSort.direction === 'asc'
                                        ? '↑'
                                        : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleValidatedFolderModuleSort('issueCount')}
                                >
                                  <span>Issues</span>
                                  {validatedFolderModuleSort?.key === 'issueCount' && (
                                    <span className="sort-indicator">
                                      {validatedFolderModuleSort.direction === 'asc'
                                        ? '↑'
                                        : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                              <th>
                                <button
                                  type="button"
                                  className="sort-button"
                                  onClick={() => toggleValidatedFolderModuleSort('reviewCount')}
                                >
                                  <span>Reviews</span>
                                  {validatedFolderModuleSort?.key === 'reviewCount' && (
                                    <span className="sort-indicator">
                                      {validatedFolderModuleSort.direction === 'asc'
                                        ? '↑'
                                        : '↓'}
                                    </span>
                                  )}
                                </button>
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {validatedFolderModuleRows.map((stat) => (
                              <tr key={stat.modulePath}>
                                <td>{stat.modulePath}</td>
                                <td>{stat.issueCount}</td>
                                <td>{stat.reviewCount}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <h2 className="table-title">Validated Issues in Folder</h2>
                      <IssueTable
                        issues={validatedSelectedFolderData.issues}
                        columns={VALIDATED_COLUMNS}
                        emptyMessage="This folder has no validated issues."
                        enableColumnPicker
                        columnPickerHint="Show or hide validated report columns for this folder."
                        refreshKey={refreshCounter}
                        defaultSort={{ key: 'priorityScore', direction: 'desc' }}
                      />
                    </>
                  ) : (
                    <p className="empty-message">
                      Select a folder to inspect validated aggregation details.
                    </p>
                  )}
                </>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  )
}

export default App
