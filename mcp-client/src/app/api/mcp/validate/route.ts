import { NextRequest } from 'next/server'
import { getMCPClient } from '@/lib/mcp-client'
import { ValidationRequest } from '@/lib/types'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json() as ValidationRequest
    
    // Validate required fields
    if (!body.rule_content) {
      return Response.json({
        error: 'Missing required field: rule_content is required'
      }, { status: 400 })
    }

    // Get MCP client and ensure connection
    const mcpClient = getMCPClient()
    await mcpClient.ensureConnected()

    // Validate business rule
    const result = await mcpClient.validateBusinessRule(body)

    return Response.json({
      success: true,
      data: result
    })

  } catch (error) {
    console.error('Business rule validation failed:', error)
    
    return Response.json({
      error: 'Failed to validate business rule',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

export async function GET() {
  return Response.json({
    endpoint: '/api/mcp/validate',
    method: 'POST',
    description: 'Validate business rules using MCP server',
    required_fields: ['rule_content'],
    optional_fields: ['examples', 'session_id', 'provider']
  })
}