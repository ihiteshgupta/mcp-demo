'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { LoadingOverlay, LoadingSpinner } from '@/components/ui/loading'
import { getMCPClient } from '@/lib/mcp-client'
import { SearchRequest, SearchResponse } from '@/lib/types'
import { generateSessionId, copyToClipboard, truncateText } from '@/lib/utils'
import { Search, Copy, FileText, Database } from 'lucide-react'

export function ContextSearch() {
  const [query, setQuery] = useState('')
  const [limit, setLimit] = useState(5)
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mcpClient = getMCPClient()

  const handleSearch = async () => {
    if (!query.trim()) {
      setError('Please enter a search query')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      await mcpClient.ensureConnected()

      const request: SearchRequest = {
        query: query.trim(),
        limit,
        session_id: generateSessionId()
      }

      const response = await mcpClient.searchContext(request)
      setSearchResults(response)
    } catch (error) {
      console.error('Search failed:', error)
      setError(error instanceof Error ? error.message : 'Failed to search context')
    } finally {
      setIsLoading(false)
    }
  }

  const handleCopy = async (content: string) => {
    try {
      await copyToClipboard(content)
    } catch (error) {
      console.error('Failed to copy:', error)
    }
  }

  const handleCopyAll = async () => {
    if (!searchResults) return
    
    const allResults = `Context Search Results

Query: "${searchResults.query}"
Total Results: ${searchResults.total_results}
Search Date: ${new Date().toLocaleString()}

Results:
${searchResults.results.map((result, index) => `
${index + 1}. Score: ${result.score.toFixed(3)}
${result.content}
Metadata: ${JSON.stringify(result.metadata, null, 2)}
`).join('\n---\n')}
`
    
    try {
      await copyToClipboard(allResults)
    } catch (error) {
      console.error('Failed to copy:', error)
    }
  }

  const resetForm = () => {
    setQuery('')
    setSearchResults(null)
    setError(null)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSearch()
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100'
    return 'text-red-600 bg-red-100'
  }

  return (
    <div className="flex flex-col space-y-4 md:space-y-6">
      <Card className="bg-gray-800/50 border-gray-700">
        <CardHeader className="pb-4 md:pb-6">
          <CardTitle className="flex items-center space-x-2 text-white">
            <Search className="w-5 h-5 text-purple-500" />
            <span className="text-lg md:text-xl">Context Search</span>
          </CardTitle>
          <CardDescription className="text-gray-400 text-sm md:text-base">
            Search for relevant business context, rules, and information using AI-powered semantic search
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col space-y-4 md:space-y-6 p-4 md:p-6">
          <LoadingOverlay isLoading={isLoading} message="Searching context...">
            {/* Search Input */}
            <div className="flex flex-col space-y-3">
              <label className="text-sm font-medium text-white">
                Search Query *
              </label>
              <div className="flex flex-col sm:flex-row gap-2">
                <Input
                  placeholder="Enter your search query..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={handleKeyPress}
                  className="flex-1 bg-gray-700 border-gray-600 text-white placeholder:text-gray-400 focus:border-blue-500 focus:ring-blue-500"
                />
                <Button 
                  onClick={handleSearch} 
                  disabled={isLoading || !query.trim()}
                  className="bg-purple-600 hover:bg-purple-700 w-full sm:w-auto"
                >
                  {isLoading ? (
                    <LoadingSpinner size="sm" />
                  ) : (
                    <Search className="w-4 h-4" />
                  )}
                </Button>
              </div>
              <p className="text-xs text-gray-400">
                Search for business rules, context, or related information
              </p>
            </div>

            {/* Search Settings */}
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="flex items-center space-x-3">
                <label className="text-sm font-medium text-white whitespace-nowrap">
                  Results Limit:
                </label>
                <select
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value))}
                  className="px-3 py-2 border border-gray-600 rounded-md text-sm bg-gray-700 text-white focus:border-blue-500 focus:ring-blue-500"
                >
                  <option value={3}>3 results</option>
                  <option value={5}>5 results</option>
                  <option value={10}>10 results</option>
                  <option value={20}>20 results</option>
                </select>
              </div>
              <Button variant="outline" size="sm" onClick={resetForm} className="border-gray-600 text-gray-300 hover:bg-gray-700">
                Reset
              </Button>
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-4 bg-red-900/20 border border-red-500/50 rounded-md text-red-300 text-sm">
                {error}
              </div>
            )}
          </LoadingOverlay>
        </CardContent>
      </Card>

      {/* Search Results */}
      {searchResults && (
        <Card className="bg-gray-800/50 border-gray-700">
          <CardHeader className="pb-4 md:pb-6">
            <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4">
              <div className="flex-1">
                <CardTitle className="flex items-center space-x-2 text-white">
                  <Database className="w-5 h-5 text-purple-500" />
                  <span className="text-lg md:text-xl">Search Results</span>
                </CardTitle>
                <CardDescription className="text-gray-400 text-sm">
                  Found {searchResults.total_results} results for "{searchResults.query}"
                </CardDescription>
              </div>
              {searchResults.results.length > 0 && (
                <Button variant="outline" size="sm" onClick={handleCopyAll} className="border-gray-600 text-gray-300 hover:bg-gray-700 w-full lg:w-auto">
                  <Copy className="w-4 h-4 mr-1" />
                  Copy All
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent className="p-4 md:p-6">
            {searchResults.results.length === 0 ? (
              <div className="text-center py-8">
                <FileText className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                <h3 className="text-lg font-medium text-white mb-2">
                  No Results Found
                </h3>
                <p className="text-gray-400">
                  Try adjusting your search query or adding more specific terms
                </p>
              </div>
            ) : (
              <div className="flex flex-col space-y-4 md:space-y-6">
                {searchResults.results.map((result, index) => (
                  <div key={index} className="border border-gray-600/50 bg-gray-700/30 rounded-lg p-4 md:p-6">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-3">
                      <div className="flex items-center space-x-3">
                        <span className="text-sm font-medium text-white">
                          Result #{index + 1}
                        </span>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          result.score >= 0.8 ? 'text-green-300 bg-green-500/20 border border-green-500/50' :
                          result.score >= 0.6 ? 'text-yellow-300 bg-yellow-500/20 border border-yellow-500/50' :
                          'text-red-300 bg-red-500/20 border border-red-500/50'
                        }`}>
                          {(result.score * 100).toFixed(1)}% match
                        </span>
                      </div>
                      <Button 
                        variant="outline" 
                        size="sm" 
                        onClick={() => handleCopy(result.content)}
                        className="border-gray-600 text-gray-300 hover:bg-gray-700 w-full sm:w-auto"
                      >
                        <Copy className="w-3 h-3 mr-1" />
                        Copy
                      </Button>
                    </div>

                    {/* Content */}
                    <div className="mb-4">
                      <div className="text-sm text-gray-300 leading-relaxed">
                        {result.content.length > 500 ? (
                          <details className="cursor-pointer">
                            <summary className="font-medium hover:text-white">
                              {truncateText(result.content, 500)}
                              <span className="text-blue-400 ml-1">(click to expand)</span>
                            </summary>
                            <div className="mt-3 pl-4 border-l-2 border-gray-600">
                              <pre className="whitespace-pre-wrap text-sm bg-gray-800/50 p-3 rounded border border-gray-600/50">
                                {result.content}
                              </pre>
                            </div>
                          </details>
                        ) : (
                          <pre className="whitespace-pre-wrap bg-gray-800/50 p-3 rounded border border-gray-600/50">
                            {result.content}
                          </pre>
                        )}
                      </div>
                    </div>

                    {/* Metadata */}
                    {Object.keys(result.metadata).length > 0 && (
                      <div className="border-t border-gray-600 pt-4">
                        <div className="text-xs text-gray-400">
                          <strong className="text-white">Metadata:</strong>
                          <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                            {Object.entries(result.metadata).map(([key, value]) => (
                              <div key={key} className="bg-gray-800/50 p-2 rounded border border-gray-600/50">
                                <span className="font-medium text-white">{key}:</span>
                                <span className="ml-1">{String(value)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Search Info */}
            {searchResults.results.length > 0 && (
              <div className="mt-6 pt-6 border-t border-gray-700">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
                  <div className="bg-gray-700/30 p-3 rounded-lg border border-gray-600/50">
                    <span className="font-medium text-white block mb-1">Query:</span>
                    <span className="text-gray-400">{searchResults.query}</span>
                  </div>
                  <div className="bg-gray-700/30 p-3 rounded-lg border border-gray-600/50">
                    <span className="font-medium text-white block mb-1">Results:</span>
                    <span className="text-gray-400">{searchResults.total_results}</span>
                  </div>
                  <div className="bg-gray-700/30 p-3 rounded-lg border border-gray-600/50">
                    <span className="font-medium text-white block mb-1">Session:</span>
                    <span className="text-gray-400">{searchResults.session_id?.slice(-8)}</span>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}