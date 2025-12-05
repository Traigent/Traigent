export type ReviewCategory = 'performance' | 'code_quality' | 'soundness_correctness'

export interface RawIssue {
  id?: unknown
  title?: unknown
  severity?: unknown
  location?: unknown
  description?: unknown
  impact?: unknown
  scope?: unknown
  evidence?: unknown
  [key: string]: unknown
}

export interface RawReviewFile {
  module?: unknown
  category?: unknown
  summary?: unknown
  issues?: unknown
  [key: string]: unknown
}

export interface IssueRecord {
  modulePath: string
  category: ReviewCategory
  id: string
  title: string
  severity: string
  location: string
  description: string
  impact: string
  scope: string[]
  sourcePath: string
  raw?: Record<string, unknown>
  priorityScore?: string
  effort?: string
  status?: string
  lineSpan?: string
  provenance?: string
}

export interface ModuleData {
  modulePath: string
  issues: IssueRecord[]
  categoryCounts: Record<ReviewCategory, number>
  reviewCount: number
}

export interface FolderModuleStat {
  modulePath: string
  issueCount: number
  reviewCount: number
}

export interface FolderData {
  folderPath: string
  issues: IssueRecord[]
  reviewCount: number
  moduleStats: FolderModuleStat[]
  reviewedModules: number
  totalModules: number
  coverage: number
}

export interface ReviewDataBundle {
  allIssues: IssueRecord[]
  issuesByCategory: Record<ReviewCategory, IssueRecord[]>
  modules: ModuleData[]
  folders: FolderData[]
}
