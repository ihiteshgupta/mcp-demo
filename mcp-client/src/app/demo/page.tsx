'use client'

import { useState, useEffect } from 'react'
import { useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { LoadingSpinner } from '@/components/ui/loading'
import { RuleGenerator } from '@/components/business-rules/RuleGenerator'
import { RuleValidator } from '@/components/business-rules/RuleValidator'
import { ContextSearch } from '@/components/business-rules/ContextSearch'
import { getMCPClient } from '@/lib/mcp-client'
import { MCPClientStatus } from '@/lib/types'
import { ArrowLeft, Sparkles, Shield, Search, Activity } from 'lucide-react'

type TabType = 'generator' | 'validation' | 'search' | 'status'

export default function DemoPage() {
  const searchParams = useSearchParams()
  const initialTab = (searchParams.get('tab') as TabType) || 'generator'
  
  const [activeTab, setActiveTab] = useState<TabType>(initialTab)
  const [mcpStatus, setMcpStatus] = useState<MCPClientStatus>({
    connected: false,
    connecting: false,
    reconnectAttempts: 0
  })
  const [isConnecting, setIsConnecting] = useState(false)

  const mcpClient = getMCPClient()

  useEffect(() => {
    // Listen for status changes
    const handleStatusChange = (status: MCPClientStatus) => {
      setMcpStatus(status)
      setIsConnecting(status.connecting)
    }

    mcpClient.on('statusChange', handleStatusChange)
    
    // Get initial status
    setMcpStatus(mcpClient.getStatus())

    // Auto-connect if not connected
    if (!mcpClient.isConnected() && !mcpClient.getStatus().connecting) {
      handleConnect()
    }

    return () => {
      mcpClient.off('statusChange', handleStatusChange)
    }
  }, [mcpClient])

  const handleConnect = async () => {
    try {
      setIsConnecting(true)
      await mcpClient.connect()
      await mcpClient.initialize()
    } catch (error) {
      console.error('Failed to connect:', error)
    } finally {
      setIsConnecting(false)
    }
  }

  const handleDisconnect = async () => {
    await mcpClient.disconnect()
  }

  const tabs = [
    {
      id: 'generator' as TabType,
      label: 'Rule Generator',
      icon: Sparkles,
      description: 'Generate new business rules'
    },
    {
      id: 'validation' as TabType,
      label: 'Rule Validator',
      icon: Shield,
      description: 'Validate existing rules'
    },
    {
      id: 'search' as TabType,
      label: 'Context Search',
      icon: Search,
      description: 'Search business context'
    },
    {
      id: 'status' as TabType,
      label: 'Status',
      icon: Activity,
      description: 'System status'
    }
  ]

  const renderTabContent = () => {
    if (!mcpStatus.connected && activeTab !== 'status') {
      return (
        <Card className="bg-gray-800/50 border-gray-700">
          <CardContent className="flex flex-col items-center justify-center py-8 md:py-12 px-4">
            <div className="text-center space-y-4 max-w-md">
              <div className="w-12 h-12 md:w-16 md:h-16 rounded-full bg-red-500/20 flex items-center justify-center mx-auto">
                <Activity className="w-6 h-6 md:w-8 md:h-8 text-red-400" />
              </div>
              <h3 className="text-base md:text-lg font-semibold text-white">
                Not Connected to MCP Server
              </h3>
              <p className="text-gray-300 text-sm md:text-base">
                Please connect to the MCP server to use the GenAI features.
              </p>
              <Button onClick={handleConnect} disabled={isConnecting} className="bg-blue-600 hover:bg-blue-700 w-full sm:w-auto">
                {isConnecting ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    Connecting...
                  </>
                ) : (
                  'Connect to Server'
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )
    }

    switch (activeTab) {
      case 'generator':
        return <RuleGenerator />
      case 'validation':
        return <RuleValidator />
      case 'search':
        return <ContextSearch />
      case 'status':
        return <StatusPanel mcpStatus={mcpStatus} onConnect={handleConnect} onDisconnect={handleDisconnect} />
      default:
        return <RuleGenerator />
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      {/* Header Section */}
      <div className="border-b border-gray-700/50 bg-gray-800/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 md:py-6">
          <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-6 w-full lg:w-auto">
              <Link href="/">
                <Button variant="ghost" className="text-gray-300 hover:text-white hover:bg-gray-700/50 px-3 py-2">
                  <ArrowLeft className="w-4 h-4 mr-2" />
                  Back to Home
                </Button>
              </Link>
              <div className="sm:border-l border-gray-600 sm:pl-6">
                <h1 className="text-xl md:text-2xl font-bold text-white mb-1">
                  MCP GenAI Demo
                </h1>
                <p className="text-gray-400 text-xs md:text-sm">
                  Interactive business rule generation and analysis
                </p>
              </div>
              <Link href="/demo/guided" className="sm:border-l border-gray-600 sm:pl-6">
                <Button variant="outline" className="border-blue-500/50 text-blue-300 hover:bg-blue-500/10 px-3 py-2">
                  <Brain className="w-4 h-4 mr-2" />
                  Guided Demo
                </Button>
              </Link>
            </div>
            
            {/* Connection Status */}
            <div className="flex items-center justify-start lg:justify-end w-full lg:w-auto">
              <div className={`flex items-center space-x-2 px-3 py-2 rounded-full border ${
                mcpStatus.connected 
                  ? 'bg-green-500/10 border-green-500/30 text-green-400' 
                  : mcpStatus.connecting 
                    ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400'
                    : 'bg-red-500/10 border-red-500/30 text-red-400'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  mcpStatus.connected ? 'bg-green-400' : 
                  mcpStatus.connecting ? 'bg-yellow-400' : 'bg-red-400'
                }`} />
                <span className="text-xs font-medium">
                  {mcpStatus.connected ? 'Connected' : 
                   mcpStatus.connecting ? 'Connecting' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="mb-6 md:mb-8">
          <div className="flex flex-col sm:flex-row space-y-1 sm:space-y-0 sm:space-x-1 bg-gray-800/50 p-1 rounded-xl border border-gray-700/50 backdrop-blur-sm">
            {tabs.map((tab) => {
              const Icon = tab.icon
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 flex items-center justify-start sm:justify-center space-x-3 px-4 sm:px-6 py-3 sm:py-4 rounded-lg text-sm font-medium transition-all duration-200 ${
                    activeTab === tab.id
                      ? 'bg-white text-gray-900 shadow-lg sm:transform sm:scale-[1.02]'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                  }`}
                >
                  <Icon className={`w-4 h-4 sm:w-5 sm:h-5 flex-shrink-0 ${activeTab === tab.id ? 'text-gray-700' : ''}`} />
                  <div className="text-left sm:text-center">
                    <div className={`font-semibold text-xs sm:text-sm ${activeTab === tab.id ? 'text-gray-900' : ''}`}>
                      {tab.label}
                    </div>
                    <div className={`text-xs hidden sm:block ${activeTab === tab.id ? 'text-gray-600' : 'text-gray-500'}`}>
                      {tab.description}
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
        </div>

        {/* Tab Content */}
        <div className="min-h-[400px] md:min-h-[600px]">
          {renderTabContent()}
        </div>
      </div>
    </div>
  )
}

// Status Panel Component
function StatusPanel({ 
  mcpStatus, 
  onConnect, 
  onDisconnect 
}: { 
  mcpStatus: MCPClientStatus
  onConnect: () => void
  onDisconnect: () => void
}) {
  return (
    <div className="space-y-6">
      <Card className="bg-gray-800/50 border-gray-700">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2 text-white">
            <Activity className="w-5 h-5 text-blue-400" />
            <span>System Status</span>
          </CardTitle>
          <CardDescription className="text-gray-400">
            Current status of MCP client and server connections
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Connection Status */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 border border-gray-600 rounded-lg bg-gray-700/30">
              <h3 className="font-medium text-white mb-2">
                MCP Server Connection
              </h3>
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${
                  mcpStatus.connected ? 'bg-green-500' : 
                  mcpStatus.connecting ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <span className="text-sm text-gray-300">
                  {mcpStatus.connected ? 'Connected and Ready' : 
                   mcpStatus.connecting ? 'Connecting...' : 'Disconnected'}
                </span>
              </div>
              {mcpStatus.error && (
                <p className="text-sm text-red-400 mt-2">{mcpStatus.error}</p>
              )}
            </div>

            <div className="p-4 border border-gray-600 rounded-lg bg-gray-700/30">
              <h3 className="font-medium text-white mb-2">
                Reconnection Attempts
              </h3>
              <p className="text-2xl font-bold text-white">
                {mcpStatus.reconnectAttempts}
              </p>
              <p className="text-sm text-gray-300">
                Automatic reconnection attempts
              </p>
            </div>
          </div>

          {/* Server Information */}
          <div className="space-y-3">
            <h3 className="font-medium text-white">
              Server Configuration
            </h3>
            <div className="bg-gray-700/30 border border-gray-600 p-4 rounded-md">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-white">Server URL:</span><br />
                  <code className="text-xs bg-gray-800 text-green-400 px-2 py-1 rounded">
                    {process.env.NEXT_PUBLIC_MCP_SERVER_URL || 'ws://localhost:8000/mcp'}
                  </code>
                </div>
                <div>
                  <span className="font-medium text-white">Protocol Version:</span><br />
                  <span className="text-gray-300">2024-11-05</span>
                </div>
                <div>
                  <span className="font-medium text-white">Client Name:</span><br />
                  <span className="text-gray-300">mcp-client</span>
                </div>
                <div>
                  <span className="font-medium text-white">Client Version:</span><br />
                  <span className="text-gray-300">1.0.0</span>
                </div>
              </div>
            </div>
          </div>

          {/* Available Features */}
          <div className="space-y-3">
            <h3 className="font-medium text-white">
              Available Features
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-md">
                <Sparkles className="w-5 h-5 text-green-400 mb-2" />
                <h4 className="font-medium text-green-300">
                  Rule Generation
                </h4>
                <p className="text-sm text-green-200/80">
                  AI-powered business rule creation
                </p>
              </div>
              <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-md">
                <Shield className="w-5 h-5 text-blue-400 mb-2" />
                <h4 className="font-medium text-blue-300">
                  Rule Validation
                </h4>
                <p className="text-sm text-blue-200/80">
                  Automated rule quality assessment
                </p>
              </div>
              <div className="p-4 bg-purple-500/10 border border-purple-500/30 rounded-md">
                <Search className="w-5 h-5 text-purple-400 mb-2" />
                <h4 className="font-medium text-purple-300">
                  Context Search
                </h4>
                <p className="text-sm text-purple-200/80">
                  Semantic business context search
                </p>
              </div>
            </div>
          </div>

          {/* Connection Controls */}
          <div className="flex space-x-3">
            {!mcpStatus.connected && !mcpStatus.connecting && (
              <Button onClick={onConnect} className="bg-blue-600 hover:bg-blue-700">
                Connect to Server
              </Button>
            )}
            {mcpStatus.connecting && (
              <Button disabled className="bg-gray-600">
                <LoadingSpinner size="sm" className="mr-2" />
                Connecting...
              </Button>
            )}
            {mcpStatus.connected && (
              <Button variant="outline" onClick={onDisconnect} className="border-gray-600 text-gray-300 hover:bg-gray-700">
                Disconnect
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}