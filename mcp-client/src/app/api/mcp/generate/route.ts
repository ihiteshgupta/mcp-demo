import { NextRequest } from 'next/server'
import { getMCPClient } from '@/lib/mcp-client'
import { BusinessRuleRequest } from '@/lib/types'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json() as BusinessRuleRequest
    
    // Validate required fields
    if (!body.context || !body.requirements) {
      return Response.json({
        error: 'Missing required fields: context and requirements are required'
      }, { status: 400 })
    }

    // Get MCP client and ensure connection
    const mcpClient = getMCPClient()
    await mcpClient.ensureConnected()

    // Generate business rule
    const result = await mcpClient.generateBusinessRule(body)

    return Response.json({
      success: true,
      data: result
    })

  } catch (error) {
    console.error('Business rule generation failed:', error)
    
    return Response.json({
      error: 'Failed to generate business rule',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 })
  }
}

export async function GET() {
  return Response.json({
    endpoint: '/api/mcp/generate',
    method: 'POST',
    description: 'Generate business rules using MCP server',
    required_fields: ['context', 'requirements'],
    optional_fields: ['rule_id', 'examples', 'session_id', 'user_id', 'provider', 'model', 'temperature']
  })
}