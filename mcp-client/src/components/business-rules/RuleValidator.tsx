'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { LoadingOverlay, LoadingSpinner } from '@/components/ui/loading'
import { getMCPClient } from '@/lib/mcp-client'
import { ValidationRequest, ValidationResponse } from '@/lib/types'
import { generateSessionId, copyToClipboard, downloadAsFile } from '@/lib/utils'
import { CheckCircle, XCircle, AlertCircle, Copy, Download, Shield } from 'lucide-react'

export function RuleValidator() {
  const [ruleContent, setRuleContent] = useState('')
  const [examples, setExamples] = useState('')
  const [provider, setProvider] = useState<'openai' | 'anthropic' | 'local' | 'lmstudio'>('lmstudio')
  const [validationResult, setValidationResult] = useState<ValidationResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mcpClient = getMCPClient()

  const handleValidate = async () => {
    if (!ruleContent.trim()) {
      setError('Please provide rule content to validate')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      await mcpClient.ensureConnected()

      const request: ValidationRequest = {
        rule_content: ruleContent.trim(),
        examples: examples.trim() ? examples.split('\n').filter(ex => ex.trim()) : undefined,
        session_id: generateSessionId(),
        provider
      }

      const response = await mcpClient.validateBusinessRule(request)
      setValidationResult(response)
    } catch (error) {
      console.error('Validation failed:', error)
      setError(error instanceof Error ? error.message : 'Failed to validate business rule')
    } finally {
      setIsLoading(false)
    }
  }

  const handleCopy = async () => {
    if (!validationResult) return
    
    const validationText = `Business Rule Validation Report

Score: ${validationResult.validation.score}/10

Strengths:
${validationResult.validation.strengths.map(s => `- ${s}`).join('\n')}

Issues Found:
${validationResult.validation.issues.map(i => `- ${i}`).join('\n')}

Recommendations:
${validationResult.validation.recommendations.map(r => `- ${r}`).join('\n')}

${validationResult.validation.revised_rule ? `
Revised Rule:
${validationResult.validation.revised_rule}
` : ''}

Implementation Notes:
${validationResult.validation.implementation_notes.map(n => `- ${n}`).join('\n')}

Validated: ${new Date().toLocaleString()}
`
    
    try {
      await copyToClipboard(validationText)
    } catch (error) {
      console.error('Failed to copy:', error)
    }
  }

  const handleDownload = () => {
    if (!validationResult) return
    
    const validationData = {
      validation: validationResult.validation,
      session_id: validationResult.session_id,
      validated_at: new Date().toISOString(),
      original_rule: ruleContent
    }
    
    downloadAsFile(
      JSON.stringify(validationData, null, 2),
      `rule_validation_${Date.now()}.json`,
      'application/json'
    )
  }

  const resetForm = () => {
    setRuleContent('')
    setExamples('')
    setValidationResult(null)
    setError(null)
  }

  const getScoreColor = (score: number) => {
    if (score >= 8) return 'text-green-600'
    if (score >= 6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getScoreBgColor = (score: number) => {
    if (score >= 8) return 'bg-green-100 border-green-200'
    if (score >= 6) return 'bg-yellow-100 border-yellow-200'
    return 'bg-red-100 border-red-200'
  }

  return (
    <div className="flex flex-col space-y-4 md:space-y-6">
      <Card className="bg-gray-800/50 border-gray-700">
        <CardHeader className="pb-4 md:pb-6">
          <CardTitle className="flex items-center space-x-2 text-white">
            <Shield className="w-5 h-5 text-green-500" />
            <span className="text-lg md:text-xl">Business Rule Validator</span>
          </CardTitle>
          <CardDescription className="text-gray-400 text-sm md:text-base">
            Validate and improve your business rules with AI-powered analysis and recommendations
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col space-y-4 md:space-y-6 p-4 md:p-6">
          <LoadingOverlay isLoading={isLoading} message="Validating business rule...">
            {/* Rule Content Input */}
            <div className="flex flex-col space-y-3">
              <label className="text-sm font-medium text-white">
                Business Rule Content *
              </label>
              <Textarea
                placeholder="Paste your business rule content here for validation..."
                value={ruleContent}
                onChange={(e) => setRuleContent(e.target.value)}
                rows={8}
                className="min-h-[200px] font-mono text-sm bg-gray-700 border-gray-600 text-white placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-400">
                Provide the complete business rule text that you want to validate
              </p>
            </div>

            {/* Validation Examples */}
            <div className="flex flex-col space-y-3">
              <label className="text-sm font-medium text-white">
                Validation Examples (Optional)
              </label>
              <Textarea
                placeholder="Provide test cases or scenarios to validate against, one per line..."
                value={examples}
                onChange={(e) => setExamples(e.target.value)}
                rows={3}
                className="min-h-[60px] bg-gray-700 border-gray-600 text-white placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-400">
                Add test scenarios or examples to validate the rule against
              </p>
            </div>

            {/* Provider Selection */}
            <div className="flex flex-col space-y-3">
              <label className="text-sm font-medium text-white">
                LLM Provider
              </label>
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value as any)}
                className="w-full px-3 py-2.5 border border-gray-600 rounded-md bg-gray-700 text-white text-sm focus:border-blue-500 focus:ring-blue-500"
              >
                <option value="lmstudio">LM Studio (Local AI)</option>
                <option value="local">Local (Demo)</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
              </select>
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-4 bg-red-900/20 border border-red-500/50 rounded-md text-red-300 text-sm">
                {error}
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-3">
              <Button 
                onClick={handleValidate} 
                disabled={isLoading || !ruleContent.trim()}
                className="flex-1 bg-green-600 hover:bg-green-700"
              >
                {isLoading ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    Validating...
                  </>
                ) : (
                  <>
                    <Shield className="w-4 h-4 mr-2" />
                    Validate Rule
                  </>
                )}
              </Button>
              <Button variant="outline" onClick={resetForm} className="border-gray-600 text-gray-300 hover:bg-gray-700">
                Reset
              </Button>
            </div>
          </LoadingOverlay>
        </CardContent>
      </Card>

      {/* Validation Results */}
      {validationResult && (
        <Card className="bg-gray-800/50 border-gray-700">
          <CardHeader className="pb-4 md:pb-6">
            <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
              <div className="flex-1">
                <CardTitle className="text-white text-lg md:text-xl">Validation Results</CardTitle>
                <CardDescription className="text-gray-400 text-sm">
                  Analysis completed on {new Date().toLocaleString()}
                </CardDescription>
              </div>
              <div className="flex flex-col sm:flex-row gap-2 w-full lg:w-auto">
                <Button variant="outline" size="sm" onClick={handleCopy} className="border-gray-600 text-gray-300 hover:bg-gray-700">
                  <Copy className="w-4 h-4 mr-1" />
                  Copy
                </Button>
                <Button variant="outline" size="sm" onClick={handleDownload} className="border-gray-600 text-gray-300 hover:bg-gray-700">
                  <Download className="w-4 h-4 mr-1" />
                  Export
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col space-y-4 md:space-y-6 p-4 md:p-6">
            {/* Validation Score */}
            <div className={`p-4 md:p-6 rounded-lg border ${
              validationResult.validation.score >= 8 ? 'bg-green-500/20 border-green-500/50' :
              validationResult.validation.score >= 6 ? 'bg-yellow-500/20 border-yellow-500/50' :
              'bg-red-500/20 border-red-500/50'
            }`}>
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="flex items-center space-x-3">
                  {validationResult.validation.score >= 8 ? (
                    <CheckCircle className="w-8 h-8 text-green-400" />
                  ) : validationResult.validation.score >= 6 ? (
                    <AlertCircle className="w-8 h-8 text-yellow-400" />
                  ) : (
                    <XCircle className="w-8 h-8 text-red-400" />
                  )}
                  <div>
                    <h3 className="text-lg font-semibold text-white">
                      Validation Score
                    </h3>
                    <p className="text-sm text-gray-300">
                      Overall rule quality assessment
                    </p>
                  </div>
                </div>
                <div className={`text-2xl md:text-3xl font-bold ${
                  validationResult.validation.score >= 8 ? 'text-green-400' :
                  validationResult.validation.score >= 6 ? 'text-yellow-400' :
                  'text-red-400'
                }`}>
                  {validationResult.validation.score}/10
                </div>
              </div>
            </div>

            {/* Strengths */}
            {validationResult.validation.strengths.length > 0 && (
              <div className="flex flex-col space-y-3">
                <h4 className="flex items-center space-x-2 font-medium text-white">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span>Strengths</span>
                </h4>
                <div className="flex flex-col space-y-3 bg-gray-700/30 p-4 rounded-lg border border-gray-600/50">
                  {validationResult.validation.strengths.map((strength, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-green-400 rounded-full mt-2 flex-shrink-0"></div>
                      <p className="text-sm text-gray-300 leading-relaxed">{strength}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Issues */}
            {validationResult.validation.issues.length > 0 && (
              <div className="flex flex-col space-y-3">
                <h4 className="flex items-center space-x-2 font-medium text-white">
                  <XCircle className="w-5 h-5 text-red-400" />
                  <span>Issues Found</span>
                </h4>
                <div className="flex flex-col space-y-3 bg-gray-700/30 p-4 rounded-lg border border-gray-600/50">
                  {validationResult.validation.issues.map((issue, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-red-400 rounded-full mt-2 flex-shrink-0"></div>
                      <p className="text-sm text-gray-300 leading-relaxed">{issue}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {validationResult.validation.recommendations.length > 0 && (
              <div className="flex flex-col space-y-3">
                <h4 className="flex items-center space-x-2 font-medium text-white">
                  <AlertCircle className="w-5 h-5 text-blue-400" />
                  <span>Recommendations</span>
                </h4>
                <div className="flex flex-col space-y-3 bg-gray-700/30 p-4 rounded-lg border border-gray-600/50">
                  {validationResult.validation.recommendations.map((recommendation, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                      <p className="text-sm text-gray-300 leading-relaxed">{recommendation}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Revised Rule */}
            {validationResult.validation.revised_rule && (
              <div className="flex flex-col space-y-3">
                <h4 className="font-medium text-white">Revised Rule</h4>
                <div className="p-4 bg-gray-700/30 rounded-lg border border-gray-600/50">
                  <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono leading-relaxed">
                    {validationResult.validation.revised_rule}
                  </pre>
                </div>
              </div>
            )}

            {/* Implementation Notes */}
            {validationResult.validation.implementation_notes.length > 0 && (
              <div className="flex flex-col space-y-3">
                <h4 className="font-medium text-white">Implementation Notes</h4>
                <div className="flex flex-col space-y-3 bg-gray-700/30 p-4 rounded-lg border border-gray-600/50">
                  {validationResult.validation.implementation_notes.map((note, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="w-2 h-2 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
                      <p className="text-sm text-gray-300 leading-relaxed">{note}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Session Info */}
            <div className="pt-6 border-t border-gray-700">
              <div className="flex flex-col sm:flex-row gap-4 text-sm">
                <div className="bg-gray-700/30 p-3 rounded-lg border border-gray-600/50">
                  <span className="font-medium text-white block mb-1">Session ID:</span>
                  <span className="text-gray-400">{validationResult.session_id?.slice(-8)}</span>
                </div>
                <div className="bg-gray-700/30 p-3 rounded-lg border border-gray-600/50">
                  <span className="font-medium text-white block mb-1">Provider:</span>
                  <span className="text-gray-400">{provider}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}