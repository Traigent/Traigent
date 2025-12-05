import type { ReactNode } from 'react'

interface TabItem {
  id: string
  label: string
  badge?: ReactNode
}

interface TabBarProps {
  tabs: TabItem[]
  activeId: string
  onSelect: (id: string) => void
}

export const TabBar = ({ tabs, activeId, onSelect }: TabBarProps) => (
  <div className="tab-bar">
    {tabs.map((tab) => (
      <button
        key={tab.id}
        type="button"
        className={`tab-button ${tab.id === activeId ? 'active' : ''}`}
        onClick={() => onSelect(tab.id)}
      >
        <span>{tab.label}</span>
        {tab.badge !== undefined && <span className="tab-badge">{tab.badge}</span>}
      </button>
    ))}
  </div>
)
