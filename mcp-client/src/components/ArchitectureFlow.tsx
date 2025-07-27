'use client'

import React, { useCallback, useMemo } from 'react'
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  Connection,
  Edge,
  BackgroundVariant,
  MarkerType,
} from 'reactflow'
import 'reactflow/dist/style.css'
import { 
  Monitor, 
  Server, 
  Database, 
  Brain, 
  Zap,
  Globe,
  MessageSquare,
  Search,
  Shield
} from 'lucide-react'

// Custom Node Component
const CustomNode = ({ data }: { data: any }) => {
  const IconComponent = data.icon
  
  return (
    <div className={`px-3 sm:px-4 md:px-6 py-2 sm:py-3 md:py-4 shadow-2xl rounded-xl border-2 ${data.bgColor} ${data.borderColor} min-w-[140px] sm:min-w-[160px] md:min-w-[180px] max-w-[200px]`}>
      <div className="flex items-center space-x-2 sm:space-x-3">
        <div className={`p-1 sm:p-2 rounded-lg ${data.iconBg}`}>
          <IconComponent className={`w-4 h-4 sm:w-5 sm:h-5 md:w-6 md:h-6 ${data.iconColor}`} />
        </div>
        <div className="min-w-0 flex-1">
          <div className={`font-bold text-xs sm:text-sm ${data.titleColor} truncate`}>{data.label}</div>
          <div className={`text-xs ${data.descColor} mt-1 truncate`}>{data.description}</div>
        </div>
      </div>
      {data.status && (
        <div className={`mt-2 sm:mt-3 px-2 py-1 rounded-full text-xs font-medium ${data.statusBg} ${data.statusColor} text-center`}>
          {data.status}
        </div>
      )}
    </div>
  )
}

const nodeTypes = {
  custom: CustomNode,
}

interface ArchitectureFlowProps {
  mcpStatus?: {
    connected: boolean
    connecting: boolean
  }
}

export function ArchitectureFlow({ mcpStatus }: ArchitectureFlowProps) {
  const isConnected = mcpStatus?.connected || false
  const isConnecting = mcpStatus?.connecting || false

  const initialNodes = useMemo(() => [
    {
      id: '1',
      type: 'custom',
      position: { x: 50, y: 100 },
      data: {
        label: 'User Interface',
        description: 'Next.js Frontend',
        icon: Monitor,
        bgColor: 'bg-gray-800/90 backdrop-blur-sm',
        borderColor: 'border-blue-500/50',
        titleColor: 'text-blue-300',
        descColor: 'text-gray-400',
        iconBg: 'bg-blue-500/20',
        iconColor: 'text-blue-400',
        status: 'Active',
        statusBg: 'bg-green-500/20',
        statusColor: 'text-green-400'
      },
    },
    {
      id: '2',
      type: 'custom',
      position: { x: 300, y: 50 },
      data: {
        label: 'WebSocket Client',
        description: 'MCP Protocol',
        icon: MessageSquare,
        bgColor: 'bg-gray-800/90 backdrop-blur-sm',
        borderColor: isConnected ? 'border-green-500/50' : isConnecting ? 'border-yellow-500/50' : 'border-red-500/50',
        titleColor: 'text-green-300',
        descColor: 'text-gray-400',
        iconBg: 'bg-green-500/20',
        iconColor: 'text-green-400',
        status: isConnected ? 'Connected' : isConnecting ? 'Connecting...' : 'Disconnected',
        statusBg: isConnected ? 'bg-green-500/20' : isConnecting ? 'bg-yellow-500/20' : 'bg-red-500/20',
        statusColor: isConnected ? 'text-green-400' : isConnecting ? 'text-yellow-400' : 'text-red-400'
      },
    },
    {
      id: '3',
      type: 'custom',
      position: { x: 600, y: 100 },
      data: {
        label: 'MCP Server',
        description: 'FastAPI Backend',
        icon: Server,
        bgColor: 'bg-gray-800/90 backdrop-blur-sm',
        borderColor: isConnected ? 'border-purple-500/50' : 'border-gray-500/50',
        titleColor: 'text-purple-300',
        descColor: 'text-gray-400',
        iconBg: 'bg-purple-500/20',
        iconColor: 'text-purple-400',
        status: isConnected ? 'Running' : 'Standby',
        statusBg: isConnected ? 'bg-purple-500/20' : 'bg-gray-500/20',
        statusColor: isConnected ? 'text-purple-400' : 'text-gray-400'
      },
    },
    {
      id: '4',
      type: 'custom',
      position: { x: 450, y: 300 },
      data: {
        label: 'LLM Provider',
        description: 'AI Engine (LM Studio)',
        icon: Brain,
        bgColor: 'bg-gray-800/90 backdrop-blur-sm',
        borderColor: 'border-pink-500/50',
        titleColor: 'text-pink-300',
        descColor: 'text-gray-400',
        iconBg: 'bg-pink-500/20',
        iconColor: 'text-pink-400',
        status: 'Local Models',
        statusBg: 'bg-pink-500/20',
        statusColor: 'text-pink-400'
      },
    },
    {
      id: '5',
      type: 'custom',
      position: { x: 750, y: 300 },
      data: {
        label: 'Redis Cache',
        description: 'Session Storage',
        icon: Database,
        bgColor: 'bg-gray-800/90 backdrop-blur-sm',
        borderColor: 'border-red-500/50',
        titleColor: 'text-red-300',
        descColor: 'text-gray-400',
        iconBg: 'bg-red-500/20',
        iconColor: 'text-red-400',
        status: 'Active',
        statusBg: 'bg-red-500/20',
        statusColor: 'text-red-400'
      },
    },
    {
      id: '6',
      type: 'custom',
      position: { x: 150, y: 300 },
      data: {
        label: 'Vector Store',
        description: 'ChromaDB',
        icon: Search,
        bgColor: 'bg-gray-800/90 backdrop-blur-sm',
        borderColor: 'border-orange-500/50',
        titleColor: 'text-orange-300',
        descColor: 'text-gray-400',
        iconBg: 'bg-orange-500/20',
        iconColor: 'text-orange-400',
        status: 'Memory Mode',
        statusBg: 'bg-orange-500/20',
        statusColor: 'text-orange-400'
      },
    },
    {
      id: '7',
      type: 'custom',
      position: { x: 300, y: 450 },
      data: {
        label: 'Business Rules',
        description: 'Rule Engine',
        icon: Shield,
        bgColor: 'bg-gray-800/90 backdrop-blur-sm',
        borderColor: 'border-cyan-500/50',
        titleColor: 'text-cyan-300',
        descColor: 'text-gray-400',
        iconBg: 'bg-cyan-500/20',
        iconColor: 'text-cyan-400',
        status: 'Ready',
        statusBg: 'bg-cyan-500/20',
        statusColor: 'text-cyan-400'
      },
    },
    {
      id: '8',
      type: 'custom',
      position: { x: 600, y: 450 },
      data: {
        label: 'LM Studio',
        description: 'OpenAI Compatible API',
        icon: Globe,
        bgColor: 'bg-gray-800/90 backdrop-blur-sm',
        borderColor: 'border-emerald-500/50',
        titleColor: 'text-emerald-300',
        descColor: 'text-gray-400',
        iconBg: 'bg-emerald-500/20',
        iconColor: 'text-emerald-400',
        status: 'Bridging',
        statusBg: 'bg-emerald-500/20',
        statusColor: 'text-emerald-400'
      },
    },
  ], [isConnected, isConnecting])

  const initialEdges = useMemo(() => [
    {
      id: 'e1-2',
      source: '1',
      target: '2',
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#3b82f6', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#3b82f6' },
    },
    {
      id: 'e2-3',
      source: '2',
      target: '3',
      type: 'smoothstep',
      animated: isConnected,
      style: { 
        stroke: isConnected ? '#10b981' : '#6b7280', 
        strokeWidth: isConnected ? 3 : 2,
        strokeDasharray: isConnected ? '0' : '5,5'
      },
      markerEnd: { 
        type: MarkerType.ArrowClosed, 
        color: isConnected ? '#10b981' : '#6b7280' 
      },
    },
    {
      id: 'e3-4',
      source: '3',
      target: '4',
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#ec4899', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#ec4899' },
    },
    {
      id: 'e3-5',
      source: '3',
      target: '5',
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#ef4444', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#ef4444' },
    },
    {
      id: 'e3-6',
      source: '3',
      target: '6',
      type: 'smoothstep',
      animated: true,
      style: { stroke: '#f97316', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#f97316' },
    },
    {
      id: 'e6-7',
      source: '6',
      target: '7',
      type: 'smoothstep',
      style: { stroke: '#06b6d4', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#06b6d4' },
    },
    {
      id: 'e4-8',
      source: '4',
      target: '8',
      type: 'smoothstep',
      style: { stroke: '#10b981', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#10b981' },
    },
    {
      id: 'e7-4',
      source: '7',
      target: '4',
      type: 'smoothstep',
      style: { stroke: '#8b5cf6', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#8b5cf6' },
    },
  ], [isConnected])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  )

  return (
    <div className="w-full h-[400px] sm:h-[500px] md:h-[600px] bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        nodeTypes={nodeTypes}
        fitView
        className="bg-gray-900"
        minZoom={0.2}
        maxZoom={1.5}
        defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
      >
        <Controls className="bg-gray-800 border-gray-600" />
        <MiniMap 
          className="bg-gray-800 border-gray-600 hidden sm:block"
          nodeColor={() => '#374151'}
          maskColor="rgba(0, 0, 0, 0.8)"
        />
        <Background 
          variant={BackgroundVariant.Dots} 
          gap={20} 
          size={1} 
          color="#374151" 
        />
      </ReactFlow>
    </div>
  )
}