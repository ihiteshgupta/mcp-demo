import { NextRequest } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    const serverUrl = process.env.MCP_SERVER_HTTP_URL || 'http://localhost:8000'
    
    // Check server health
    const response = await fetch(`${serverUrl}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    })

    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`)
    }

    const serverHealth = await response.json()

    return Response.json({
      client: {
        status: 'healthy',
        timestamp: new Date().toISOString()
      },
      server: {
        status: 'healthy',
        url: serverUrl,
        health: serverHealth
      },
      mcp: {
        websocket_url: process.env.NEXT_PUBLIC_MCP_SERVER_URL || 'ws://localhost:8000/mcp',
        protocol_version: '2024-11-05'
      }
    })

  } catch (error) {
    console.error('MCP status check failed:', error)
    
    return Response.json({
      client: {
        status: 'healthy',
        timestamp: new Date().toISOString()
      },
      server: {
        status: 'unhealthy',
        url: process.env.MCP_SERVER_HTTP_URL || 'http://localhost:8000',
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      mcp: {
        websocket_url: process.env.NEXT_PUBLIC_MCP_SERVER_URL || 'ws://localhost:8000/mcp',
        protocol_version: '2024-11-05'
      }
    }, { status: 503 })
  }
}