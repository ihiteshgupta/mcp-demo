'use client'

import {useState, useEffect} from 'react'
import Link from 'next/link'
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from '@/components/ui/card'
import {Button} from '@/components/ui/button'
import {LoadingSpinner} from '@/components/ui/loading'
import {ArchitectureFlow} from '@/components/ArchitectureFlow'
import {getMCPClient} from '@/lib/mcp-client'
import {MCPClientStatus} from '@/lib/types'
import {
    Server,
    Zap,
    Brain,
    Shield,
    Search,
    Network,
    Database,
    Cpu,
    CheckCircle,
    AlertCircle,
    XCircle,
    Loader2,
    ExternalLink,
    Play,
    Settings,
    Activity,
    Sparkles,
    ArrowRight,
    Globe
} from 'lucide-react'

export default function HomePage() {
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

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
            {/* Hero Section */}
            <div className="relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600/20 via-purple-600/20 to-pink-600/20"/>
                <div className="relative">
                    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-24">
                        {/* Header */}
                        <div className="text-center mb-16">
                            <div className="flex items-center justify-center mb-8">
                                <div className="relative">
                                    <div className="absolute inset-0 bg-blue-500/20 rounded-full blur-xl scale-150"/>
                                    <div
                                        className="relative p-4 bg-blue-500/10 backdrop-blur-sm rounded-full border border-blue-500/30">
                                        <Brain className="w-12 h-12 text-blue-400"/>
                                    </div>
                                </div>
                            </div>
                            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold text-white mb-4 md:mb-6 tracking-tight px-4">
                                MCP GenAI Demo
                            </h1>
                            <p className="text-lg sm:text-xl md:text-2xl text-gray-300 max-w-4xl mx-auto leading-relaxed px-4">
                                Experience the power of <span className="text-blue-400 font-semibold">Model Context Protocol</span> with
                                intelligent business rule generation powered by local AI models
                            </p>

                            {/* Quick Stats */}
                            <div
                                className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6 mt-8 md:mt-12 max-w-3xl mx-auto px-4">
                                <div className="text-center p-2">
                                    <div className="text-xl sm:text-2xl font-bold text-blue-400">100%</div>
                                    <div className="text-xs sm:text-sm text-gray-400">Local AI</div>
                                </div>
                                <div className="text-center p-2">
                                    <div className="text-xl sm:text-2xl font-bold text-green-400">Real-time</div>
                                    <div className="text-xs sm:text-sm text-gray-400">Processing</div>
                                </div>
                                <div className="text-center p-2">
                                    <div className="text-xl sm:text-2xl font-bold text-purple-400">No API</div>
                                    <div className="text-xs sm:text-sm text-gray-400">Keys Required</div>
                                </div>
                                <div className="text-center p-2">
                                    <div className="text-xl sm:text-2xl font-bold text-pink-400">Open</div>
                                    <div className="text-xs sm:text-sm text-gray-400">Source</div>
                                </div>
                            </div>
                        </div>

                        {/* Status Card */}
                        <Card
                            className="bg-gray-800/60 backdrop-blur-sm border-gray-700/50 shadow-2xl mb-8 md:mb-16 mx-4">
                            <CardContent className="p-4 md:p-8">
                                <div
                                    className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                                    <div className="flex items-center space-x-4">
                                        <div className={`p-3 rounded-full ${
                                            mcpStatus.connected ? 'bg-green-500/20' :
                                                mcpStatus.connecting ? 'bg-yellow-500/20' : 'bg-red-500/20'
                                        }`}>
                                            {mcpStatus.connected ? (
                                                <CheckCircle className="w-6 h-6 text-green-400"/>
                                            ) : mcpStatus.connecting ? (
                                                <Loader2 className="w-6 h-6 text-yellow-400 animate-spin"/>
                                            ) : (
                                                <XCircle className="w-6 h-6 text-red-400"/>
                                            )}
                                        </div>
                                        <div>
                                            <h3 className="text-base md:text-lg font-semibold text-white">
                                                {mcpStatus.connected ? 'System Connected' :
                                                    mcpStatus.connecting ? 'Connecting to System...' :
                                                        'System Disconnected'}
                                            </h3>
                                            <p className="text-gray-400 text-xs md:text-sm">
                                                {mcpStatus.connected ? 'All components are running and ready' :
                                                    mcpStatus.connecting ? 'Establishing WebSocket connection...' :
                                                        'Click to connect to the MCP server'}
                                            </p>
                                        </div>
                                    </div>
                                    <div
                                        className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3 w-full sm:w-auto">
                                        {!mcpStatus.connected && !mcpStatus.connecting && (
                                            <Button
                                                onClick={handleConnect}
                                                disabled={isConnecting}
                                                size="lg"
                                                className="bg-blue-600 hover:bg-blue-700 text-white px-4 md:px-8 w-full sm:w-auto"
                                            >
                                                {isConnecting ? (
                                                    <>
                                                        <Loader2 className="w-5 h-5 mr-2 animate-spin"/>
                                                        Connecting...
                                                    </>
                                                ) : (
                                                    <>
                                                        <Play className="w-5 h-5 mr-2"/>
                                                        Start System
                                                    </>
                                                )}
                                            </Button>
                                        )}
                                        {mcpStatus.connected && (
                                            <div
                                                className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3 w-full sm:w-auto">
                                                <Link href="/demo" className="w-full sm:w-auto">
                                                    <Button size="lg"
                                                            className="bg-green-600 hover:bg-green-700 px-4 md:px-8 w-full sm:w-auto">
                                                        <Sparkles className="w-5 h-5 mr-2"/>
                                                        Launch Demo
                                                        <ArrowRight className="w-4 h-4 ml-2"/>
                                                    </Button>
                                                </Link>
                                                <Link href="/demo/guided" className="w-full sm:w-auto">
                                                    <Button size="lg"
                                                            variant="outline"
                                                            className="border-blue-500/50 text-blue-300 hover:bg-blue-500/10 w-full sm:w-auto">
                                                        <Brain className="w-5 h-5 mr-2"/>
                                                        Guided Demo
                                                    </Button>
                                                </Link>
                                                <Button
                                                    variant="outline"
                                                    onClick={handleDisconnect}
                                                    className="border-gray-600 text-gray-300 hover:bg-gray-700 w-full sm:w-auto"
                                                >
                                                    Disconnect
                                                </Button>
                                            </div>
                                        )}
                                    </div>
                                </div>
                                {mcpStatus.error && (
                                    <div
                                        className="mt-6 p-4 bg-red-900/20 border border-red-700/50 rounded-lg text-red-300 text-sm flex items-center">
                                        <AlertCircle className="w-5 h-5 mr-3 flex-shrink-0"/>
                                        <span>{mcpStatus.error}</span>
                                    </div>
                                )}
                            </CardContent>
                        </Card>

                        {/* Architecture Section */}
                        <Card className="bg-gray-800/60 backdrop-blur-sm border-gray-700/50 shadow-2xl mx-4">
                            <CardHeader className="pb-6 md:pb-8">
                                <div className="text-center">
                                    <div
                                        className="flex flex-col sm:flex-row items-center justify-center space-y-2 sm:space-y-0 sm:space-x-3 mb-4">
                                        <Activity className="w-6 h-6 md:w-8 md:h-8 text-blue-400"/>
                                        <CardTitle className="text-xl md:text-2xl lg:text-3xl font-bold text-white">System
                                            Architecture</CardTitle>
                                    </div>
                                    <CardDescription
                                        className="text-sm md:text-base lg:text-lg text-gray-400 max-w-2xl mx-auto px-4 mb-6">
                                        Interactive visualization of the MCP GenAI Demo architecture and real-time data
                                        flow
                                    </CardDescription>

                                    {/* Connection Flow Legend */}
                                    <div className="flex flex-wrap justify-center gap-4 text-xs text-gray-400">
                                        <div className="flex items-center space-x-2">
                                            <div className="w-3 h-0.5 bg-blue-500"></div>
                                            <span>User Interface</span>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <div
                                                className={`w-3 h-0.5 ${mcpStatus.connected ? 'bg-green-500' : 'bg-gray-500'}`}></div>
                                            <span>MCP Protocol</span>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-3 h-0.5 bg-pink-500"></div>
                                            <span>AI Processing</span>
                                        </div>
                                        <div className="flex items-center space-x-2">
                                            <div className="w-3 h-0.5 bg-orange-500"></div>
                                            <span>Data Storage</span>
                                        </div>
                                    </div>
                                </div>
                            </CardHeader>
                            <CardContent className="p-4 md:p-6 lg:p-8">
                                <div className="mb-8">
                                    <ArchitectureFlow mcpStatus={mcpStatus}/>
                                </div>

                                {/* Technology Stack */}
                                <div
                                    className="mt-6 md:mt-12 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
                                    <div
                                        className="text-center p-4 md:p-6 bg-gray-700/30 rounded-xl border border-gray-600/50">
                                        <Globe className="w-6 h-6 md:w-8 md:h-8 text-blue-400 mx-auto mb-2 md:mb-3"/>
                                        <h4 className="font-semibold text-blue-300 mb-1 md:mb-2 text-sm md:text-base">Frontend</h4>
                                        <p className="text-xs md:text-sm text-gray-400">Next.js 14, React 18,
                                            TypeScript, Tailwind CSS</p>
                                    </div>
                                    <div
                                        className="text-center p-4 md:p-6 bg-gray-700/30 rounded-xl border border-gray-600/50">
                                        <Server className="w-6 h-6 md:w-8 md:h-8 text-purple-400 mx-auto mb-2 md:mb-3"/>
                                        <h4 className="font-semibold text-purple-300 mb-1 md:mb-2 text-sm md:text-base">Backend</h4>
                                        <p className="text-xs md:text-sm text-gray-400">FastAPI, Python, WebSocket, MCP
                                            Protocol</p>
                                    </div>
                                    <div
                                        className="text-center p-4 md:p-6 bg-gray-700/30 rounded-xl border border-gray-600/50">
                                        <Brain className="w-6 h-6 md:w-8 md:h-8 text-pink-400 mx-auto mb-2 md:mb-3"/>
                                        <h4 className="font-semibold text-pink-300 mb-1 md:mb-2 text-sm md:text-base">AI
                                            Engine</h4>
                                        <p className="text-xs md:text-sm text-gray-400">LM Studio, Local Models,
                                            No API Keys</p>
                                    </div>
                                    <div
                                        className="text-center p-4 md:p-6 bg-gray-700/30 rounded-xl border border-gray-600/50">
                                        <Database
                                            className="w-6 h-6 md:w-8 md:h-8 text-green-400 mx-auto mb-2 md:mb-3"/>
                                        <h4 className="font-semibold text-green-300 mb-1 md:mb-2 text-sm md:text-base">Storage</h4>
                                        <p className="text-xs md:text-sm text-gray-400">Redis Cache, ChromaDB Vectors,
                                            Memory Store</p>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                        {/* Features Grid */}
                        <div
                            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6 lg:gap-8 mb-8 md:mb-16 px-4 mt-3">
                            <Card
                                className="group hover:scale-105 transition-all duration-300 bg-gradient-to-br from-green-500/10 to-green-600/5 border-green-500/30 hover:border-green-400/50">
                                <CardContent className="p-4 md:p-6 lg:p-8">
                                    <div
                                        className="flex flex-col sm:flex-row items-start sm:items-center space-y-3 sm:space-y-0 sm:space-x-4 mb-4 md:mb-6">
                                        <div
                                            className="p-2 md:p-3 bg-green-500/20 rounded-xl group-hover:bg-green-500/30 transition-colors">
                                            <Brain className="w-6 h-6 md:w-8 md:h-8 text-green-400"/>
                                        </div>
                                        <h3 className="text-lg md:text-xl font-bold text-white">Business Rule
                                            Generator</h3>
                                    </div>
                                    <p className="text-gray-300 mb-4 md:mb-6 leading-relaxed text-sm md:text-base">
                                        Generate intelligent business rules using AI-powered analysis with natural
                                        language processing
                                    </p>
                                    <Link href="/demo">
                                        <Button
                                            className="w-full bg-green-600 hover:bg-green-700 group-hover:shadow-lg group-hover:shadow-green-500/25"
                                            disabled={!mcpStatus.connected}
                                            size="lg"
                                        >
                                            <Brain className="w-5 h-5 mr-2"/>
                                            Generate Rules
                                        </Button>
                                    </Link>
                                </CardContent>
                            </Card>

                            <Card
                                className="group hover:scale-105 transition-all duration-300 bg-gradient-to-br from-purple-500/10 to-purple-600/5 border-purple-500/30 hover:border-purple-400/50">
                                <CardContent className="p-4 md:p-6 lg:p-8">
                                    <div
                                        className="flex flex-col sm:flex-row items-start sm:items-center space-y-3 sm:space-y-0 sm:space-x-4 mb-4 md:mb-6">
                                        <div
                                            className="p-2 md:p-3 bg-purple-500/20 rounded-xl group-hover:bg-purple-500/30 transition-colors">
                                            <Shield className="w-6 h-6 md:w-8 md:h-8 text-purple-400"/>
                                        </div>
                                        <h3 className="text-lg md:text-xl font-bold text-white">Rule Validation</h3>
                                    </div>
                                    <p className="text-gray-300 mb-4 md:mb-6 leading-relaxed text-sm md:text-base">
                                        Validate and improve existing business rules with AI feedback and quality
                                        assessment
                                    </p>
                                    <Link href="/demo?tab=validation">
                                        <Button
                                            variant="outline"
                                            className="w-full border-purple-500/50 text-purple-300 hover:bg-purple-500/10 group-hover:shadow-lg group-hover:shadow-purple-500/25"
                                            disabled={!mcpStatus.connected}
                                            size="lg"
                                        >
                                            <Shield className="w-5 h-5 mr-2"/>
                                            Validate Rules
                                        </Button>
                                    </Link>
                                </CardContent>
                            </Card>

                            <Card
                                className="group hover:scale-105 transition-all duration-300 bg-gradient-to-br from-orange-500/10 to-orange-600/5 border-orange-500/30 hover:border-orange-400/50">
                                <CardContent className="p-4 md:p-6 lg:p-8">
                                    <div
                                        className="flex flex-col sm:flex-row items-start sm:items-center space-y-3 sm:space-y-0 sm:space-x-4 mb-4 md:mb-6">
                                        <div
                                            className="p-2 md:p-3 bg-orange-500/20 rounded-xl group-hover:bg-orange-500/30 transition-colors">
                                            <Search className="w-6 h-6 md:w-8 md:h-8 text-orange-400"/>
                                        </div>
                                        <h3 className="text-lg md:text-xl font-bold text-white">Context Search</h3>
                                    </div>
                                    <p className="text-gray-300 mb-4 md:mb-6 leading-relaxed text-sm md:text-base">
                                        Search and analyze relevant business context using semantic vector search
                                        capabilities
                                    </p>
                                    <Link href="/demo?tab=search">
                                        <Button
                                            variant="outline"
                                            className="w-full border-orange-500/50 text-orange-300 hover:bg-orange-500/10 group-hover:shadow-lg group-hover:shadow-orange-500/25"
                                            disabled={!mcpStatus.connected}
                                            size="lg"
                                        >
                                            <Search className="w-5 h-5 mr-2"/>
                                            Search Context
                                        </Button>
                                    </Link>
                                </CardContent>
                            </Card>
                        </div>
                    </div>
                </div>
            </div>


        </div>
    )
}