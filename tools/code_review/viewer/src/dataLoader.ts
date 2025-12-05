import type {
  FolderData,
  FolderModuleStat,
  IssueRecord,
  ModuleData,
  RawIssue,
  RawReviewFile,
  ReviewCategory,
  ReviewDataBundle,
} from './types'

const REVIEW_CATEGORIES: ReviewCategory[] = [
  'performance',
  'code_quality',
  'soundness_correctness',
]

const reviewFiles = import.meta.glob('../../out/**/*.review.json', { eager: true })
const validatedMetadataFiles = import.meta.glob(
  '../../reports/development/code_debt/**/metadata.json',
  { eager: true },
)
const pythonModules = import.meta.glob('../../traigent/**/*.py', {
  import: 'default',
  eager: false,
  query: '?url',
})

const initialCategoryCounts = (): Record<ReviewCategory, number> => ({
  performance: 0,
  code_quality: 0,
  soundness_correctness: 0,
})

const initialCategoryIssues = (): Record<ReviewCategory, IssueRecord[]> => ({
  performance: [],
  code_quality: [],
  soundness_correctness: [],
})

const isReviewCategory = (value: unknown): value is ReviewCategory =>
  typeof value === 'string' && REVIEW_CATEGORIES.includes(value as ReviewCategory)

const sanitizeString = (value: unknown, fallback = ''): string => {
  if (typeof value === 'string') return value
  if (value === null || value === undefined) return fallback
  try {
    return String(value)
  } catch {
    return fallback
  }
}

const sanitizeScope = (value: unknown): string[] => {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => sanitizeString(item).trim())
    .filter((entry) => entry.length > 0)
}

const deriveLocation = (issue: RawIssue, modulePath: string): string => {
  const locationRaw = sanitizeString(issue.location, '').trim()
  if (locationRaw) return locationRaw

  const evidence = sanitizeString(issue.evidence, '')
  const evidenceMatch = evidence.match(/([A-Za-z0-9_./-]+\.py)(?::|#L?)(\d+)/)
  if (evidenceMatch) {
    const [, filePath, line] = evidenceMatch
    return `${filePath}:${line}`
  }

  const lineMatch = evidence.match(/line\s+(\d+)/i)
  if (lineMatch) {
    const [, line] = lineMatch
    return `${modulePath}:${line}`
  }

  const scope = sanitizeScope(issue.scope)
  if (scope.length > 0) {
    return `${modulePath}#${scope[0]}`
  }

  return 'n/a'
}

const normalizeIssue = (
  issue: RawIssue,
  modulePath: string,
  category: ReviewCategory,
  sourcePath: string,
): IssueRecord => ({
  modulePath,
  category,
  id: sanitizeString(issue.id, 'Unassigned'),
  title: sanitizeString(issue.title, 'Untitled issue'),
  severity: sanitizeString(issue.severity, 'unspecified'),
  location: deriveLocation(issue, modulePath),
  description: sanitizeString(issue.description, ''),
  impact: sanitizeString(issue.impact, ''),
  scope: sanitizeScope(issue.scope),
  sourcePath,
  raw: issue as Record<string, unknown>,
})

const toRawReview = (input: unknown): RawReviewFile | null =>
  input && typeof input === 'object' ? (input as RawReviewFile) : null

const getFolderPath = (modulePath: string): string => {
  const parts = modulePath.split('/').filter(Boolean)
  if (parts.length <= 1) return '(root)'
  parts.pop()
  return parts.join('/')
}

const normalizeModulePath = (rawPath: string): string =>
  rawPath.replace(/^(\.\.\/)+/, '').replace(/^\/+/, '')

const shouldCountPythonModule = (modulePath: string): boolean =>
  modulePath.endsWith('.py') &&
  !modulePath.endsWith('__init__.py') &&
  !modulePath.includes('/tests/')

const formatProvenance = (value: unknown): string => {
  if (!Array.isArray(value)) return ''
  return (value as unknown[])
    .map((entry) => {
      if (entry && typeof entry === 'object') {
        const path = sanitizeString((entry as Record<string, unknown>).path, '')
        const ruleId = sanitizeString((entry as Record<string, unknown>).rule_id, '')
        if (path && ruleId) return `${path} (${ruleId})`
        return path || ruleId
      }
      return sanitizeString(entry, '')
    })
    .filter((item) => item.length > 0)
    .join('; ')
}

const pythonFolderCounts: Map<string, number> = (() => {
  const counts = new Map<string, number>()
  for (const rawPath of Object.keys(pythonModules)) {
    const normalizedPath = normalizeModulePath(rawPath)
    if (!shouldCountPythonModule(normalizedPath)) continue
    const folderPath = getFolderPath(normalizedPath)
    counts.set(folderPath, (counts.get(folderPath) ?? 0) + 1)
  }
  return counts
})()

const decorateFolders = (
  folders: Array<{
    folderPath: string
    issues: IssueRecord[]
    reviewCount: number
    moduleStats: FolderModuleStat[]
  }>,
): FolderData[] =>
  folders
    .map(({ folderPath, issues, reviewCount, moduleStats }) => {
      const reviewedModules = moduleStats.length
      const totalModules =
        pythonFolderCounts.get(folderPath) ??
        (moduleStats.length > 0 ? moduleStats.length : 0)
      const coverage =
        totalModules === 0 ? (reviewedModules > 0 ? 1 : 0) : Math.min(1, reviewedModules / totalModules)
      return {
        folderPath,
        issues,
        reviewCount,
        moduleStats,
        reviewedModules,
        totalModules,
        coverage,
      }
    })
    .sort((a, b) => a.folderPath.localeCompare(b.folderPath))

export const loadReviewData = (): ReviewDataBundle => {
  const issuesByCategory = initialCategoryIssues()
  const allIssues: IssueRecord[] = []
  const modulesMap = new Map<string, ModuleData>()

  for (const [sourcePath, moduleExport] of Object.entries(reviewFiles)) {
    const payload = (moduleExport as { default?: unknown })?.default ?? moduleExport
    const rawReview = toRawReview(payload)
    if (!rawReview) continue

    const modulePath = sanitizeString(rawReview.module, '').trim()
    if (!modulePath) continue

    const categoryRaw = rawReview.category
    if (!isReviewCategory(categoryRaw)) continue
    const category = categoryRaw

    const rawIssues = Array.isArray(rawReview.issues)
      ? (rawReview.issues as RawIssue[])
      : []

    let moduleRecord = modulesMap.get(modulePath)
    if (!moduleRecord) {
      moduleRecord = {
        modulePath,
        issues: [],
        categoryCounts: initialCategoryCounts(),
        reviewCount: 0,
      }
      modulesMap.set(modulePath, moduleRecord)
    }

    moduleRecord.reviewCount += 1

    const normalizedIssues = rawIssues.map((issue) =>
      normalizeIssue(issue, modulePath, category, sourcePath),
    )

    moduleRecord.issues.push(...normalizedIssues)
    moduleRecord.categoryCounts[category] += normalizedIssues.length

    issuesByCategory[category].push(...normalizedIssues)
    allIssues.push(...normalizedIssues)
  }

  const modules: ModuleData[] = Array.from(modulesMap.values()).sort((a, b) =>
    a.modulePath.localeCompare(b.modulePath),
  )

  const folderMap = new Map<string, { issues: IssueRecord[]; reviewCount: number; moduleStats: Map<string, FolderModuleStat> }>()

  for (const module of modules) {
    const folderPath = getFolderPath(module.modulePath)
    let folderRecord = folderMap.get(folderPath)
    if (!folderRecord) {
      folderRecord = {
        issues: [],
        reviewCount: 0,
        moduleStats: new Map(),
      }
      folderMap.set(folderPath, folderRecord)
    }

    folderRecord.issues.push(...module.issues)
    folderRecord.reviewCount += module.reviewCount
      folderRecord.moduleStats.set(module.modulePath, {
        modulePath: module.modulePath,
        issueCount: module.issues.length,
        reviewCount: module.reviewCount,
      })
  }

  return {
    allIssues,
    issuesByCategory,
    modules,
    folders: decorateFolders(
      Array.from(folderMap.entries()).map(([folderPath, data]) => ({
        folderPath,
        issues: [...data.issues],
        reviewCount: data.reviewCount,
        moduleStats: Array.from(data.moduleStats.values()).sort((a, b) =>
          a.modulePath.localeCompare(b.modulePath),
        ),
      })),
    ),
  }
}

export const loadValidatedData = (): ReviewDataBundle => {
  const issuesByCategory = initialCategoryIssues()
  const allIssues: IssueRecord[] = []
  const modulesMap = new Map<string, ModuleData>()

  for (const [sourcePath, moduleExport] of Object.entries(validatedMetadataFiles)) {
    const payload = (moduleExport as { default?: unknown })?.default ?? moduleExport
    if (!payload || typeof payload !== 'object') continue

    const rawMetadata = payload as {
      file?: unknown
      issues?: unknown
    }

    const modulePath = sanitizeString(rawMetadata.file, '').trim()
    if (!modulePath) continue

    const rawIssues = Array.isArray(rawMetadata.issues)
      ? (rawMetadata.issues as Record<string, unknown>[])
      : []

    let moduleRecord = modulesMap.get(modulePath)
    if (!moduleRecord) {
      moduleRecord = {
        modulePath,
        issues: [],
        categoryCounts: initialCategoryCounts(),
        reviewCount: 0,
      }
      modulesMap.set(modulePath, moduleRecord)
    }

    moduleRecord.reviewCount += rawIssues.length

    const normalizedIssues = rawIssues
      .map((rawIssue) => {
        const categoryRaw = sanitizeString(rawIssue.category, '')
        if (!isReviewCategory(categoryRaw)) return null
        const category = categoryRaw

        const issueRecord: IssueRecord = {
          modulePath,
          category,
          id: sanitizeString(rawIssue.issue_id ?? rawIssue.id, 'Unassigned'),
          title: sanitizeString(rawIssue.title, 'Untitled issue'),
          severity: sanitizeString(
            rawIssue.normalized_severity ?? rawIssue.severity,
            'unspecified',
          ),
          location: sanitizeString(rawIssue.line_span, 'n/a'),
          description: '',
          impact: '',
          scope: [],
          sourcePath,
          priorityScore: sanitizeString(rawIssue.priority_score, ''),
          effort: sanitizeString(rawIssue.effort, ''),
          status: sanitizeString(rawIssue.status, ''),
          lineSpan: sanitizeString(rawIssue.line_span, ''),
          provenance: formatProvenance(rawIssue.provenance),
          raw: rawIssue,
        }
        return issueRecord
      })
      .filter((issue): issue is IssueRecord => issue !== null)

    moduleRecord.issues.push(...normalizedIssues)
    for (const issue of normalizedIssues) {
      moduleRecord.categoryCounts[issue.category] += 1
      issuesByCategory[issue.category].push(issue)
      allIssues.push(issue)
    }
  }

  const modules: ModuleData[] = Array.from(modulesMap.values()).sort((a, b) =>
    a.modulePath.localeCompare(b.modulePath),
  )

  const folderMap = new Map<string, { issues: IssueRecord[]; reviewCount: number; moduleStats: Map<string, FolderModuleStat> }>()

  for (const module of modules) {
    const folderPath = getFolderPath(module.modulePath)
    let folderRecord = folderMap.get(folderPath)
    if (!folderRecord) {
      folderRecord = {
        issues: [],
        reviewCount: 0,
        moduleStats: new Map(),
      }
      folderMap.set(folderPath, folderRecord)
    }

    folderRecord.issues.push(...module.issues)
    folderRecord.reviewCount += module.reviewCount
    folderRecord.moduleStats.set(module.modulePath, {
      modulePath: module.modulePath,
      issueCount: module.issues.length,
      reviewCount: module.reviewCount,
    })
  }

  return {
    allIssues,
    issuesByCategory,
    modules,
    folders: decorateFolders(
      Array.from(folderMap.entries()).map(([folderPath, data]) => ({
        folderPath,
        issues: [...data.issues],
        reviewCount: data.reviewCount,
        moduleStats: Array.from(data.moduleStats.values()).sort((a, b) =>
          a.modulePath.localeCompare(b.modulePath),
        ),
      })),
    ),
  }
}
