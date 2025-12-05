import type { IssueRecord } from '../types'

const formatIssue = (issue: IssueRecord): string =>
  JSON.stringify(
    {
      modulePath: issue.modulePath,
      category: issue.category,
      id: issue.id,
      title: issue.title,
      severity: issue.severity,
      location: issue.location,
      description: issue.description,
      impact: issue.impact,
      scope: issue.scope,
  priorityScore: issue.priorityScore,
  effort: issue.effort,
  status: issue.status,
  lineSpan: issue.lineSpan,
  provenance: issue.provenance,
  sourcePath: issue.sourcePath,
  raw: issue.raw,
},
    null,
    2,
  )

export const copyIssueToClipboard = async (issue: IssueRecord): Promise<void> => {
  const payload = formatIssue(issue)
  if (typeof navigator !== 'undefined' && navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(payload)
    return
  }

  const textarea = document.createElement('textarea')
  textarea.value = payload
  textarea.setAttribute('readonly', '')
  textarea.style.position = 'absolute'
  textarea.style.left = '-9999px'
  document.body.appendChild(textarea)
  textarea.select()
  document.execCommand('copy')
  document.body.removeChild(textarea)
}
