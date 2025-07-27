'use client';

import React, { useState, useEffect } from 'react';
import { Activity, Brain, Database, MessageSquare, Zap, CheckCircle, AlertCircle, Clock } from 'lucide-react';

interface ComponentStatus {
  name: string;
  status: 'active' | 'processing' | 'idle' | 'error';
  message: string;
  lastUpdate: Date;
  metrics?: {
    responseTime?: number;
    requestCount?: number;
    successRate?: number;
  };
}

interface ThinkingStep {
  stepNumber: number;
  description: string;
  status: 'pending' | 'processing' | 'completed';
  result?: string;
  timing?: number;
}

interface LiveDashboardProps {
  autoDemo?: boolean;
}

export default function LiveDashboard({ autoDemo = false }: LiveDashboardProps) {
  const [components, setComponents] = useState<ComponentStatus[]>([
    {
      name: 'LM Studio',
      status: 'idle',
      message: 'Ready for requests',
      lastUpdate: new Date(),
      metrics: { responseTime: 0, requestCount: 0, successRate: 100 }
    },
    {
      name: 'MCP Server',
      status: 'idle',
      message: 'Protocol ready',
      lastUpdate: new Date(),
      metrics: { responseTime: 0, requestCount: 0, successRate: 100 }
    },
    {
      name: 'Prompt Builder',
      status: 'idle',
      message: 'Templates loaded',
      lastUpdate: new Date(),
      metrics: { responseTime: 0, requestCount: 0, successRate: 100 }
    },
    {
      name: 'Vector Store',
      status: 'idle',
      message: 'Memory ready',
      lastUpdate: new Date(),
      metrics: { responseTime: 0, requestCount: 0, successRate: 100 }
    },
    {
      name: 'Session Storage',
      status: 'idle',
      message: 'Sessions active',
      lastUpdate: new Date(),
      metrics: { responseTime: 0, requestCount: 0, successRate: 100 }
    }
  ]);

  const [thinkingSteps, setThinkingSteps] = useState<ThinkingStep[]>([]);
  const [currentChain, setCurrentChain] = useState<string>('');
  const [isRunning, setIsRunning] = useState(false);
  const [demoProgress, setDemoProgress] = useState(0);
  const [isMounted, setIsMounted] = useState(false);

  // Auto-demo scenarios
  const demoScenarios = [
    {
      title: 'Enterprise Volume Discount Rules',
      context: 'Telecom provider needs volume-based pricing for business customers',
      requirements: 'Create rules for 10-24 lines (15% discount) and 25+ lines (20% discount)'
    },
    {
      title: 'Seasonal Campaign Rules',
      context: 'Q4 business acquisition campaign for mid-market companies',
      requirements: 'Holiday promotion with bundle discounts and priority support'
    },
    {
      title: 'Customer Retention Rules',
      context: 'At-risk customer retention with churn prediction integration',
      requirements: 'Personalized offers based on usage patterns and contract terms'
    }
  ];

  const getStatusIcon = (status: ComponentStatus['status']) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'processing':
        return <Activity className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <Clock className="w-4 h-4 text-gray-500" />;
    }
  };

  const getComponentIcon = (name: string) => {
    switch (name) {
      case 'LM Studio':
        return <Brain className="w-5 h-5" />;
      case 'MCP Server':
        return <MessageSquare className="w-5 h-5" />;
      case 'Prompt Builder':
        return <Zap className="w-5 h-5" />;
      case 'Vector Store':
      case 'Session Storage':
        return <Database className="w-5 h-5" />;
      default:
        return <Activity className="w-5 h-5" />;
    }
  };

  const updateComponentStatus = (name: string, status: ComponentStatus['status'], message: string, timing?: number) => {
    setComponents(prev => prev.map(comp => {
      if (comp.name === name) {
        const newMetrics = { ...comp.metrics };
        if (timing) newMetrics.responseTime = timing;
        if (status === 'active') newMetrics.requestCount = (newMetrics.requestCount || 0) + 1;
        
        return {
          ...comp,
          status,
          message,
          lastUpdate: new Date(),
          metrics: newMetrics
        };
      }
      return comp;
    }));
  };

  const simulateThinkingProcess = (steps: ThinkingStep[]) => {
    setThinkingSteps(steps);
    
    steps.forEach((step, index) => {
      setTimeout(() => {
        setThinkingSteps(prev => prev.map(s => 
          s.stepNumber === step.stepNumber 
            ? { ...s, status: 'processing' as const }
            : s
        ));
        
        // Simulate processing time
        setTimeout(() => {
          setThinkingSteps(prev => prev.map(s => 
            s.stepNumber === step.stepNumber 
              ? { 
                  ...s, 
                  status: 'completed' as const,
                  result: `Completed: ${step.description}`,
                  timing: Math.random() * 1000 + 500
                }
              : s
          ));
        }, Math.random() * 2000 + 1000);
      }, index * 3000);
    });
  };

  const runAutomatedDemo = async (scenarioIndex: number) => {
    setIsRunning(true);
    setDemoProgress((scenarioIndex / demoScenarios.length) * 100);
    
    const scenario = demoScenarios[scenarioIndex];
    setCurrentChain(`Demo Chain ${scenarioIndex + 1}: ${scenario.title}`);

    // Initialize thinking steps
    const steps: ThinkingStep[] = [
      { stepNumber: 1, description: 'Analyze Business Context', status: 'pending' },
      { stepNumber: 2, description: 'Parse Requirements', status: 'pending' },
      { stepNumber: 3, description: 'Design Rule Logic', status: 'pending' },
      { stepNumber: 4, description: 'Optimize for Business Value', status: 'pending' },
      { stepNumber: 5, description: 'Format as JSON Structure', status: 'pending' }
    ];

    // Start component interactions
    updateComponentStatus('MCP Server', 'processing', 'Receiving generation request');
    
    setTimeout(() => {
      updateComponentStatus('Prompt Builder', 'processing', 'Building sequential thinking prompt');
      simulateThinkingProcess(steps);
    }, 500);

    setTimeout(() => {
      updateComponentStatus('LM Studio', 'processing', 'Generating with sequential reasoning', 1200);
    }, 2000);

    setTimeout(() => {
      updateComponentStatus('Vector Store', 'processing', 'Storing generated rule');
    }, 8000);

    setTimeout(() => {
      updateComponentStatus('Session Storage', 'processing', 'Updating session context');
    }, 9000);

    setTimeout(() => {
      // Complete all components
      updateComponentStatus('MCP Server', 'active', 'Request completed successfully');
      updateComponentStatus('Prompt Builder', 'active', 'Template cache optimized');
      updateComponentStatus('LM Studio', 'active', 'Model ready for next request');
      updateComponentStatus('Vector Store', 'active', 'Rule indexed and searchable');
      updateComponentStatus('Session Storage', 'active', 'Context preserved');
      
      // Move to next scenario or complete
      if (scenarioIndex < demoScenarios.length - 1) {
        setTimeout(() => runAutomatedDemo(scenarioIndex + 1), 3000);
      } else {
        setIsRunning(false);
        setDemoProgress(100);
      }
    }, 10000);
  };

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    if (autoDemo && !isRunning) {
      // Start auto demo after a brief delay
      setTimeout(() => runAutomatedDemo(0), 2000);
    }
  }, [autoDemo]);

  // Simulate periodic health checks
  useEffect(() => {
    const interval = setInterval(() => {
      if (!isRunning) {
        setComponents(prev => prev.map(comp => ({
          ...comp,
          lastUpdate: new Date()
        })));
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <div className="p-6 space-y-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Activity className="w-6 h-6 text-blue-600" />
            Live Component Dashboard
          </h1>
          
          {autoDemo && (
            <div className="flex items-center gap-4">
              <div className="text-sm text-gray-600">
                Demo Progress: {Math.round(demoProgress)}%
              </div>
              <div className="w-32 bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${demoProgress}%` }}
                />
              </div>
            </div>
          )}
        </div>

        {/* Component Status Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {components.map((component) => (
            <div key={component.name} className="bg-gray-50 rounded-lg p-4 border">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {getComponentIcon(component.name)}
                  <h3 className="font-semibold text-gray-900">{component.name}</h3>
                </div>
                {getStatusIcon(component.status)}
              </div>
              
              <p className="text-sm text-gray-600 mb-3">{component.message}</p>
              
              {component.metrics && (
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <div className="text-gray-500">Response</div>
                    <div className="font-mono">{component.metrics.responseTime?.toFixed(0) || 0}ms</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Requests</div>
                    <div className="font-mono">{component.metrics.requestCount || 0}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">Success</div>
                    <div className="font-mono">{component.metrics.successRate || 100}%</div>
                  </div>
                </div>
              )}
              
              <div className="text-xs text-gray-400 mt-2">
                Updated: {isMounted ? component.lastUpdate.toLocaleTimeString() : '--:--:--'}
              </div>
            </div>
          ))}
        </div>

        {/* Sequential Thinking Visualization */}
        {currentChain && (
          <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
            <h2 className="text-lg font-semibold text-blue-900 mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5" />
              {currentChain}
            </h2>
            
            <div className="space-y-3">
              {thinkingSteps.map((step) => (
                <div key={step.stepNumber} className="flex items-center gap-3 p-3 bg-white rounded border">
                  <div className="flex-shrink-0">
                    {step.status === 'completed' && <CheckCircle className="w-5 h-5 text-green-500" />}
                    {step.status === 'processing' && <Activity className="w-5 h-5 text-blue-500 animate-spin" />}
                    {step.status === 'pending' && <Clock className="w-5 h-5 text-gray-400" />}
                  </div>
                  
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">
                      Step {step.stepNumber}: {step.description}
                    </div>
                    {step.result && (
                      <div className="text-sm text-gray-600 mt-1">{step.result}</div>
                    )}
                    {step.timing && (
                      <div className="text-xs text-blue-600 mt-1">
                        Completed in {step.timing.toFixed(0)}ms
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Demo Controls */}
        <div className="bg-white rounded-lg border p-4 mt-6">
          <h3 className="font-semibold text-gray-900 mb-4">Demo Controls</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {demoScenarios.map((scenario, index) => (
              <button
                key={index}
                onClick={() => runAutomatedDemo(index)}
                disabled={isRunning}
                className="p-3 text-left border rounded-lg hover:bg-blue-50 hover:border-blue-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="font-medium text-gray-900">{scenario.title}</div>
                <div className="text-sm text-gray-600 mt-1">{scenario.context}</div>
              </button>
            ))}
          </div>
          
          <div className="mt-4 pt-4 border-t">
            <div className="text-sm text-gray-600">
              🎯 <strong>Auto Demo Mode:</strong> Watch live component interactions and sequential thinking in action
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}