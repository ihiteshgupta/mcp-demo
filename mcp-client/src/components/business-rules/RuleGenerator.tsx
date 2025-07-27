'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { LoadingOverlay, LoadingSpinner } from '@/components/ui/loading'
import { getMCPClient } from '@/lib/mcp-client'
import { BusinessRuleRequest, BusinessRuleResponse } from '@/lib/types'
import { generateSessionId, copyToClipboard, downloadAsFile } from '@/lib/utils'
import { Copy, Download, Sparkles, Zap, RotateCcw } from 'lucide-react'

// Demo scenarios for prepopulation
const demoScenarios = [
  {
    id: 'enterprise-volume',
    title: 'Enterprise Volume Discounts',
    description: 'Telecom B2B customer pricing optimization',
    context: 'Enterprise telecommunications provider serving B2B customers with tiered service offerings. Need intelligent pricing rules for Q4 customer acquisition campaign targeting mid-market companies (50-500 employees). Current challenges include competitive pricing pressure, customer churn in the 25% range, and need to optimize revenue per customer while maintaining service quality.',
    requirements: 'Create business rules for volume-based discounts:\n• 5-9 lines: 10% discount + free setup\n• 10-24 lines: 15% discount + free setup + priority support\n• 25+ lines: 20% discount + free setup + priority support + dedicated account manager\n\nRules must consider:\n• Customer tier (Startup, Growth, Enterprise)\n• Seasonal budget cycles (Q4 budget flush)\n• Competitive landscape positioning\n• Revenue optimization targets (minimum 15% margin)\n• Service bundling opportunities\n\nOutput in JSON format with sequential reasoning steps.',
    examples: 'Previous successful rule: {"when": {"lines": {"$gte": 10}, "customer_type": "enterprise"}, "then": {"discount": 15, "benefits": ["priority_support"]}}'
  },
  {
    id: 'seasonal-campaign', 
    title: 'Q4 Seasonal Campaigns',
    description: 'Holiday business acquisition campaigns',
    context: 'Telecommunications company planning Q4 business acquisition campaigns for different verticals: retail, healthcare, finance, manufacturing. Each vertical has unique needs and promotion sensitivity. Retail needs bandwidth surge capability for holiday shopping, healthcare requires compliance features, finance demands security packages, and manufacturing needs IoT connectivity.',
    requirements: 'Generate seasonal business campaign rules:\n• Industry-specific promotions (retail holiday surge, healthcare compliance, finance security)\n• Time-based activation (Black Friday, Cyber Monday, end of fiscal year)\n• Geographic targeting for regional business customers\n• Service-specific offers (bandwidth upgrades for retail, security packages for finance)\n• Budget-based tiering for different company sizes\n• Bundle discounts for multi-service packages\n• Priority support during peak periods\n\nInclude automatic campaign start/end dates and ROI tracking mechanisms.',
    examples: 'Campaign rule example: {"when": {"industry": "retail", "period": "holiday_season"}, "then": {"bandwidth_boost": "2x", "discount": 25, "duration": "Nov-Jan"}}'
  },
  {
    id: 'customer-retention',
    title: 'Customer Retention Rules', 
    description: 'At-risk customer retention with churn prediction',
    context: 'Telecommunications provider with customer lifecycle management needs. Churn prediction system identifies customers with >70% likelihood of cancellation within 90 days. Different promotional strategies required for new, existing, and churning business customers based on usage patterns, payment history, support ticket volume, and contract terms.',
    requirements: 'Create customer retention rules for at-risk customers:\n• Churn probability >70%: Immediate intervention with 25% discount + service credits\n• Churn probability 50-70%: Proactive offers with upgrade incentives\n• Payment issues: Flexible payment plans + temporary service credits\n• Low usage patterns: Service tier optimization + cost reduction\n• High support tickets: Service quality improvements + compensation\n• Contract renewal: Loyalty rewards + early renewal discounts\n\nPersonalized offers based on:\n• Usage patterns and service utilization\n• Payment history and creditworthiness\n• Support interaction history\n• Competitor threat analysis\n• Win-back campaign automation',
    examples: 'Retention rule: {"when": {"churn_risk": "high", "usage": "declining"}, "then": {"offer": "service_optimization", "discount": 20, "follow_up": "automated"}}'
  }
];

const defaultScenario = demoScenarios[0];

export function RuleGenerator() {
  const [context, setContext] = useState(defaultScenario.context)
  const [requirements, setRequirements] = useState(defaultScenario.requirements)
  const [examples, setExamples] = useState(defaultScenario.examples)
  const [provider, setProvider] = useState<'openai' | 'anthropic' | 'local' | 'lmstudio'>('lmstudio')
  const [temperature, setTemperature] = useState(0.7)
  const [generatedRule, setGeneratedRule] = useState<BusinessRuleResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedScenario, setSelectedScenario] = useState(defaultScenario.id)

  const mcpClient = getMCPClient()

  const loadScenario = (scenarioId: string) => {
    const scenario = demoScenarios.find(s => s.id === scenarioId)
    if (scenario) {
      setContext(scenario.context)
      setRequirements(scenario.requirements)
      setExamples(scenario.examples)
      setSelectedScenario(scenarioId)
      setGeneratedRule(null) // Clear previous results
      setError(null)
    }
  }

  const clearFields = () => {
    setContext('')
    setRequirements('')
    setExamples('')
    setSelectedScenario('')
    setGeneratedRule(null)
    setError(null)
  }

  const handleGenerate = async () => {
    if (!context.trim() || !requirements.trim()) {
      setError('Please provide both context and requirements')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      await mcpClient.ensureConnected()

      const request: BusinessRuleRequest = {
        context: context.trim(),
        requirements: requirements.trim(),
        examples: examples.trim() ? examples.split('\n').filter(ex => ex.trim()) : undefined,
        session_id: generateSessionId(),
        provider,
        temperature,
        metadata: {
          use_sequential_thinking: true,
          demo_scenario: selectedScenario,
          demo_mode: true
        }
      }

      const response = await mcpClient.generateBusinessRule(request)
      setGeneratedRule(response)
    } catch (error) {
      console.error('Generation failed:', error)
      setError(error instanceof Error ? error.message : 'Failed to generate business rule')
    } finally {
      setIsLoading(false)
    }
  }

  const handleCopy = async () => {
    if (!generatedRule) return
    
    const ruleText = `Business Rule: ${generatedRule.rule.name}

ID: ${generatedRule.rule.id}
Priority: ${generatedRule.rule.priority}

Description:
${generatedRule.rule.description}

Condition:
${generatedRule.rule.condition}

Action:
${generatedRule.rule.action}

Business Value:
${generatedRule.rule.business_value}

Examples:
${generatedRule.rule.examples.map(ex => `- ${ex}`).join('\n')}

Generated: ${new Date(generatedRule.rule.created_at).toLocaleString()}
Model: ${generatedRule.generation_info.model}
`
    
    try {
      await copyToClipboard(ruleText)
    } catch (error) {
      console.error('Failed to copy:', error)
    }
  }

  const handleDownload = () => {
    if (!generatedRule) return
    
    const ruleData = {
      rule: generatedRule.rule,
      generation_info: generatedRule.generation_info,
      exported_at: new Date().toISOString()
    }
    
    downloadAsFile(
      JSON.stringify(ruleData, null, 2),
      `business_rule_${generatedRule.rule.id}.json`,
      'application/json'
    )
  }

  const resetForm = () => {
    setContext('')
    setRequirements('')
    setExamples('')
    setGeneratedRule(null)
    setError(null)
  }

  return (
    <div className="space-y-4 md:space-y-6">
      <Card className="bg-gray-800/50 border-gray-700">
        <CardHeader className="pb-4 md:pb-6">
          <CardTitle className="flex items-center space-x-2 text-white">
            <Sparkles className="w-5 h-5 text-blue-500" />
            <span className="text-lg md:text-xl">Business Rule Generator</span>
          </CardTitle>
          <CardDescription className="text-gray-400 text-sm md:text-base">
            Generate intelligent business rules using AI-powered analysis based on your context and requirements
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 md:space-y-6 p-4 md:p-6">
          {/* Demo Scenario Selection */}
          <div className="space-y-4 p-4 bg-blue-900/20 border border-blue-700/30 rounded-lg">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-blue-300 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Demo Scenarios (Click to load & customize)
              </h3>
              <Button
                onClick={clearFields}
                variant="outline"
                size="sm"
                className="border-gray-600 text-gray-300 hover:bg-gray-700 h-8 px-3"
              >
                <RotateCcw className="w-3 h-3 mr-1" />
                Clear
              </Button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {demoScenarios.map((scenario) => (
                <button
                  key={scenario.id}
                  onClick={() => loadScenario(scenario.id)}
                  className={`p-3 text-left border rounded-lg transition-all ${
                    selectedScenario === scenario.id
                      ? 'border-blue-500 bg-blue-900/30 text-blue-100'
                      : 'border-gray-600 bg-gray-800/50 text-gray-300 hover:border-gray-500 hover:bg-gray-700/50'
                  }`}
                >
                  <div className="font-medium text-sm mb-1">{scenario.title}</div>
                  <div className="text-xs opacity-80">{scenario.description}</div>
                </button>
              ))}
            </div>
            <div className="text-xs text-gray-400 bg-gray-800/30 p-2 rounded border-l-4 border-blue-500">
              💡 <strong>Demo Mode:</strong> Fields are prepopulated with realistic telecom scenarios. Edit any field to customize for your use case.
            </div>
          </div>

          <LoadingOverlay isLoading={isLoading} message="Generating business rule with sequential thinking...">
            {/* Context Input */}
            <div className="space-y-3">
              <label className="text-sm font-medium text-white">
                Business Context *
              </label>
              <Textarea
                placeholder="Describe the business context, domain, and background information that will help generate relevant rules..."
                value={context}
                onChange={(e) => setContext(e.target.value)}
                rows={4}
                className="min-h-[100px] bg-gray-700 border-gray-600 text-white placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-400">
                Provide detailed context about your business domain, processes, and constraints
              </p>
            </div>

            {/* Requirements Input */}
            <div className="space-y-3">
              <label className="text-sm font-medium text-white">
                Rule Requirements *
              </label>
              <Textarea
                placeholder="Specify the exact requirements for the business rule you want to generate..."
                value={requirements}
                onChange={(e) => setRequirements(e.target.value)}
                rows={3}
                className="min-h-[80px] bg-gray-700 border-gray-600 text-white placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-400">
                Be specific about what the rule should accomplish and any constraints
              </p>
            </div>

            {/* Examples Input */}
            <div className="space-y-3">
              <label className="text-sm font-medium text-white">
                Reference Examples (Optional)
              </label>
              <Textarea
                placeholder="Provide example rules or scenarios, one per line..."
                value={examples}
                onChange={(e) => setExamples(e.target.value)}
                rows={3}
                className="min-h-[60px] bg-gray-700 border-gray-600 text-white placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-400">
                Add one example per line to guide the rule generation
              </p>
            </div>

            {/* Settings */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
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
              <div className="space-y-3">
                <label className="text-sm font-medium text-white">
                  Temperature: {temperature}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer slider"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-2">
                  <span>Conservative</span>
                  <span>Creative</span>
                </div>
              </div>
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
                onClick={handleGenerate} 
                disabled={isLoading || !context.trim() || !requirements.trim()}
                className="flex-1 bg-blue-600 hover:bg-blue-700"
              >
                {isLoading ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Generate Rule
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

      {/* Generated Rule Display */}
      {generatedRule && (
        <Card className="bg-gray-800/50 border-gray-700">
          <CardHeader className="pb-4 md:pb-6">
            <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
              <div className="flex-1">
                <CardTitle className="text-white text-lg md:text-xl">Generated Business Rule</CardTitle>
                <CardDescription className="text-gray-400 text-sm">
                  Rule ID: {generatedRule.rule.id} | 
                  Generated: {new Date(generatedRule.rule.created_at).toLocaleString()}
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
          <CardContent className="space-y-4 md:space-y-6 p-4 md:p-6">
            {/* Rule Name and Priority */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
              <h3 className="text-lg md:text-xl font-semibold text-white">
                {generatedRule.rule.name}
              </h3>
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                generatedRule.rule.priority === 'HIGH' ? 'bg-red-500/20 text-red-300 border border-red-500/50' :
                generatedRule.rule.priority === 'MEDIUM' ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/50' :
                'bg-green-500/20 text-green-300 border border-green-500/50'
              }`}>
                {generatedRule.rule.priority} Priority
              </span>
            </div>

            {/* Description */}
            <div className="space-y-3">
              <h4 className="font-medium text-white">Description</h4>
              <p className="text-gray-300 leading-relaxed bg-gray-700/30 p-4 rounded-lg border border-gray-600/50">
                {generatedRule.rule.description}
              </p>
            </div>

            {/* Condition and Action */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
              <div className="space-y-2">
                <h4 className="font-medium text-white">Condition</h4>
                <div className="p-3 bg-blue-900/20 border border-blue-500/30 rounded-md">
                  <p className="text-sm text-blue-100">
                    {generatedRule.rule.condition}
                  </p>
                </div>
              </div>
              <div className="space-y-2">
                <h4 className="font-medium text-white">Action</h4>
                <div className="p-3 bg-green-900/20 border border-green-500/30 rounded-md">
                  <p className="text-sm text-green-100">
                    {generatedRule.rule.action}
                  </p>
                </div>
              </div>
            </div>

            {/* Business Value */}
            <div className="space-y-3">
              <h4 className="font-medium text-white">Business Value</h4>
              <p className="text-gray-300 bg-gray-700/30 p-4 rounded-lg border border-gray-600/50">
                {generatedRule.rule.business_value}
              </p>
            </div>

            {/* Examples */}
            {generatedRule.rule.examples.length > 0 && (
              <div className="space-y-3">
                <h4 className="font-medium text-white">Examples</h4>
                <ul className="space-y-3 bg-gray-700/30 p-4 rounded-lg border border-gray-600/50">
                  {generatedRule.rule.examples.map((example, index) => (
                    <li key={index} className="flex items-start space-x-3">
                      <span className="text-blue-400 text-sm mt-1 font-bold">•</span>
                      <span className="text-sm text-gray-300 leading-relaxed">
                        {example}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Generation Info */}
            <div className="pt-6 border-t border-gray-700">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 text-sm">
                <div className="bg-gray-700/30 p-3 rounded-lg border border-gray-600/50">
                  <span className="font-medium text-white block mb-1">Model:</span>
                  <span className="text-gray-400">{generatedRule.generation_info.model}</span>
                </div>
                <div className="bg-gray-700/30 p-3 rounded-lg border border-gray-600/50">
                  <span className="font-medium text-white block mb-1">Finish Reason:</span>
                  <span className="text-gray-400">{generatedRule.generation_info.finish_reason}</span>
                </div>
                <div className="bg-gray-700/30 p-3 rounded-lg border border-gray-600/50">
                  <span className="font-medium text-white block mb-1">Prompt Length:</span>
                  <span className="text-gray-400">{generatedRule.generation_info.prompt_length}</span>
                </div>
                <div className="bg-gray-700/30 p-3 rounded-lg border border-gray-600/50">
                  <span className="font-medium text-white block mb-1">Session:</span>
                  <span className="text-gray-400">{generatedRule.session_id?.slice(-8)}</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}