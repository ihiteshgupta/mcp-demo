import { NextRequest } from 'next/server'
import { getMCPClient } from '@/lib/mcp-client'
import { SearchRequest } from '@/lib/types'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json() as SearchRequest
    
    // Validate required fields
    if (!body.query) {
      return Response.json({
        error: 'Missing required field: query is required'
      }, { status: 400 })
    }

    // Validate limit if provided
    if (body.limit && (body.limit < 1 || body.limit > 20)) {
      return Response.json({
        error: 'Invalid limit: must be between 1 and 20'
      }, { status: 400 })
    }

    // Get MCP client and ensure connection
    const mcpClient = getMCPClient()
    await mcpClient.ensureConnected()

    // Search context
    const result = await mcpClient.searchContext(body)

    return Response.json({
      success: true,
      data: result
    })

  } catch (error) {
    console.error('Context search failed:', error)
    
    return Response.json({
      error: 'Failed to search context',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const query = searchParams.get('query')
    const limit = parseInt(searchParams.get('limit') || '5')

    if (!query) {
      return Response.json({
        endpoint: '/api/mcp/search',
        methods: ['GET', 'POST'],
        description: 'Search business context using MCP server',
        required_fields: ['query'],
        optional_fields: ['limit', 'filter_metadata', 'session_id'],
        query_params: 'For GET requests: ?query=<search_term>&limit=<number>'
      })
    }

    // Validate limit
    if (limit < 1 || limit > 20) {
      return Response.json({
        error: 'Invalid limit: must be between 1 and 20'
      }, { status: 400 })
    }

    // Get MCP client and ensure connection
    const mcpClient = getMCPClient()
    await mcpClient.ensureConnected()

    // Search context
    const result = await mcpClient.searchContext({ query, limit })

    return Response.json({
      success: true,
      data: result
    })

  } catch (error) {
    console.error('Context search failed:', error)
    
    return Response.json({
      error: 'Failed to search context',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}