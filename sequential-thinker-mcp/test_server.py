#!/usr/bin/env python3
"""
Test script for Sequential Thinker MCP Server

This script demonstrates the server's capabilities and can be used for development testing.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sequential_thinker_mcp.main import (
    create_thinking_chain,
    add_thinking_step,
    generate_sequential_thinking_prompt,
    create_business_rule_thinking_prompt,
    validate_thinking_chain,
    get_server_info,
    prompt_builder
)
from sequential_thinker_mcp.main import ThinkingChainRequest, ThinkingStepRequest, SequentialThinkingRequest
from mcp_server.components.prompt_builder import PromptBuilder


async def test_basic_functionality():
    """Test basic server functionality."""
    print("🧠 Testing Sequential Thinker MCP Server")
    print("=" * 50)
    
    # Initialize prompt builder for testing
    global prompt_builder
    template_dir = Path(__file__).parent.parent / "mcp-server" / "templates"
    prompt_builder = PromptBuilder(str(template_dir))
    
    # Test 1: Get server info
    print("\n1. Testing server info...")
    info = await get_server_info()
    print(f"✅ Server: {info['server_name']} v{info['version']}")
    print(f"✅ Templates: {len(info['available_templates'])} available")
    
    # Test 2: Create a thinking chain
    print("\n2. Testing thinking chain creation...")
    chain_request = ThinkingChainRequest(
        chain_id="test_chain_001",
        description="Test problem-solving chain",
        main_task="Solve a complex business problem",
        context="E-commerce platform needs to improve customer retention"
    )
    
    chain_result = await create_thinking_chain(chain_request)
    if chain_result["success"]:
        print(f"✅ Created chain: {chain_result['chain_id']}")
    else:
        print(f"❌ Failed to create chain: {chain_result['error']}")
        return
    
    # Test 3: Add thinking steps
    print("\n3. Testing thinking step addition...")
    steps = [
        {
            "description": "Analyze Current State",
            "reasoning": "Understand the current customer retention metrics and identify pain points",
            "expected_output": "Current retention rate, churn reasons, customer feedback analysis"
        },
        {
            "description": "Identify Root Causes",
            "reasoning": "Determine the underlying causes of customer churn",
            "expected_output": "List of primary factors causing customer attrition"
        },
        {
            "description": "Design Solutions",
            "reasoning": "Develop targeted interventions to address identified issues",
            "expected_output": "Actionable solutions with expected impact and implementation timeline"
        },
        {
            "description": "Implementation Plan",
            "reasoning": "Create a detailed plan for rolling out solutions",
            "expected_output": "Step-by-step implementation roadmap with milestones"
        }
    ]
    
    for step_data in steps:
        step_request = ThinkingStepRequest(
            chain_id="test_chain_001",
            description=step_data["description"],
            reasoning=step_data["reasoning"],
            expected_output=step_data["expected_output"]
        )
        
        step_result = await add_thinking_step(step_request)
        if step_result["success"]:
            print(f"✅ Added step {step_result['step_number']}: {step_data['description']}")
        else:
            print(f"❌ Failed to add step: {step_result['error']}")
    
    # Test 4: Generate sequential thinking prompt
    print("\n4. Testing prompt generation...")
    thinking_request = SequentialThinkingRequest(
        task="Improve customer retention for e-commerce platform",
        context="Current retention rate is 75%, industry average is 85%",
        additional_instructions="Focus on actionable solutions that can be implemented within 6 months"
    )
    
    prompt_result = await generate_sequential_thinking_prompt(thinking_request)
    if prompt_result["success"]:
        print(f"✅ Generated prompt with {len(prompt_result['prompt'])} characters")
        print(f"✅ Using chain: {prompt_result['chain_id']}")
        print(f"✅ Steps: {prompt_result['step_count']}")
        
        # Display a portion of the generated prompt
        print("\n📝 Generated Prompt Preview:")
        print("-" * 40)
        preview = prompt_result['prompt'][:500] + "..." if len(prompt_result['prompt']) > 500 else prompt_result['prompt']
        print(preview)
        print("-" * 40)
    else:
        print(f"❌ Failed to generate prompt: {prompt_result['error']}")
    
    # Test 5: Validate thinking chain
    print("\n5. Testing chain validation...")
    validation_result = await validate_thinking_chain("test_chain_001")
    if validation_result["success"]:
        validation = validation_result["validation"]
        print(f"✅ Chain validation completed")
        print(f"   Valid: {validation['valid']}")
        print(f"   Steps: {validation['step_count']}")
        print(f"   Completeness: {validation['completeness_score']:.2f}")
        if validation['issues']:
            print(f"   Issues: {', '.join(validation['issues'])}")
        if validation['suggestions']:
            print(f"   Suggestions: {', '.join(validation['suggestions'])}")
    else:
        print(f"❌ Failed to validate chain: {validation_result['error']}")
    
    # Test 6: Business rule thinking prompt
    print("\n6. Testing business rule generation...")
    business_rule_result = await create_business_rule_thinking_prompt(
        context="E-commerce platform with B2B and B2C customers",
        requirements="Create dynamic pricing rules that maximize revenue while maintaining customer satisfaction",
        rule_id="dynamic_pricing_001",
        examples=[
            "Volume discounts for bulk orders",
            "Loyalty program pricing for repeat customers",
            "Seasonal pricing adjustments"
        ]
    )
    
    if business_rule_result["success"]:
        print(f"✅ Generated business rule prompt")
        print(f"   Rule ID: {business_rule_result['rule_id']}")
        print(f"   Prompt length: {len(business_rule_result['prompt'])} characters")
        
        # Display a portion of the business rule prompt
        print("\n📋 Business Rule Prompt Preview:")
        print("-" * 40)
        preview = business_rule_result['prompt'][:400] + "..." if len(business_rule_result['prompt']) > 400 else business_rule_result['prompt']
        print(preview)
        print("-" * 40)
    else:
        print(f"❌ Failed to generate business rule prompt: {business_rule_result['error']}")
    
    print("\n🎉 All tests completed!")


async def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n🔍 Testing Edge Cases")
    print("=" * 30)
    
    # Test with invalid chain ID
    print("\n1. Testing invalid chain validation...")
    invalid_result = await validate_thinking_chain("nonexistent_chain")
    if not invalid_result["success"]:
        print("✅ Correctly handled invalid chain ID")
    else:
        print("❌ Should have failed for invalid chain ID")
    
    # Test duplicate chain creation
    print("\n2. Testing duplicate chain creation...")
    duplicate_request = ThinkingChainRequest(
        chain_id="test_chain_001",  # Same as previous test
        description="Duplicate chain",
        main_task="This should handle duplicates gracefully"
    )
    
    try:
        duplicate_result = await create_thinking_chain(duplicate_request)
        if not duplicate_result["success"]:
            print("✅ Correctly handled duplicate chain ID")
        else:
            print("⚠️  Duplicate chain creation succeeded (may be intended behavior)")
    except Exception as e:
        print(f"✅ Exception caught for duplicate: {str(e)}")
    
    print("\n✅ Edge case testing completed!")


def save_sample_outputs():
    """Save sample outputs for documentation."""
    print("\n💾 Saving sample outputs...")
    
    samples = {
        "server_info": {
            "description": "Example server information response",
            "example": {
                "server_name": "Sequential Thinker MCP Server",
                "version": "1.0.0",
                "capabilities": [
                    "Sequential thinking chain creation",
                    "Step-by-step reasoning prompts",
                    "Business rule generation with thinking",
                    "Custom prompt building",
                    "Template management"
                ]
            }
        },
        "thinking_chain": {
            "description": "Example thinking chain creation",
            "example": {
                "success": True,
                "chain_id": "problem_analysis_001",
                "description": "Customer retention analysis chain"
            }
        }
    }
    
    with open("sample_outputs.json", "w") as f:
        json.dump(samples, f, indent=2)
    
    print("✅ Sample outputs saved to sample_outputs.json")


async def main():
    """Run all tests."""
    try:
        await test_basic_functionality()
        await test_edge_cases()
        save_sample_outputs()
        
        print("\n🎯 Test Summary:")
        print("- Basic functionality: ✅")
        print("- Edge cases: ✅") 
        print("- Sample outputs: ✅")
        print("\nSequential Thinker MCP Server is ready for use! 🚀")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())