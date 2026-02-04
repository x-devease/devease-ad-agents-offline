#!/usr/bin/env python3
"""
Ad Miner Agents - Moprobo Evolution Simulation

Run autonomous agents on real moprobo data to improve ad creative performance.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from meta.ad.miner.agents import AgentOrchestrator
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def load_moprobo_metrics():
    """Load actual moprobo ad_miner results."""
    patterns_file = Path('/Users/anthony/coding/devease-ad-agents-offline/results/moprobo/meta/ad_miner/patterns.yaml')

    with open(patterns_file, 'r') as f:
        data = yaml.safe_load(f)

    metadata = data['metadata']
    combinatorial_patterns = data.get('combinatorial_patterns', [])
    individual_features = data.get('individual_features', [])

    # Extract current performance metrics
    data_quality = metadata.get('data_quality', {})
    metrics = {
        'avg_roas': data_quality.get('avg_roas', metadata.get('avg_roas', 31.65)),
        'top_quartile_roas': data_quality.get('top_quartile_roas', metadata.get('top_quartile_roas', 34.18)),
        'bottom_quartile_roas': data_quality.get('bottom_quartile_roas', metadata.get('bottom_quartile_roas', 11.51)),
        'roas_range': data_quality.get('roas_range', metadata.get('roas_range', 1953.89)),
        'sample_size': metadata.get('sample_size', 1187),
        'top_quartile_size': metadata.get('top_quartile_size', 297),
        'bottom_quartile_size': metadata.get('bottom_quartile_size', 297),
        'data_quality': data_quality.get('completeness_score', 0.85),
    }

    # Extract top patterns
    top_patterns = combinatorial_patterns[:3] if combinatorial_patterns else []

    return {
        'metrics': metrics,
        'top_patterns': top_patterns,
        'customer': metadata['customer'],
        'product': metadata['product'],
    }

def main():
    print('\n' + '='*80)
    print('AD MINER AGENTS - MOPROBO EVOLUTION SIMULATION')
    print('='*80)

    # Load real moprobo data
    print('\n📊 Loading real moprobo data...')
    moprobo_data = load_moprobo_metrics()

    print(f'\n  Customer: {moprobo_data["customer"]}')
    print(f'  Product: {moprobo_data["product"]}')
    print(f'\n  Current Performance:')
    print(f'    - Average ROAS: {moprobo_data["metrics"]["avg_roas"]:.2f}x')
    print(f'    - Top Quartile ROAS: {moprobo_data["metrics"]["top_quartile_roas"]:.2f}x')
    print(f'    - Bottom Quartile ROAS: {moprobo_data["metrics"]["bottom_quartile_roas"]:.2f}x')
    print(f'    - ROAS Range: {moprobo_data["metrics"]["roas_range"]:.2f}x')
    print(f'    - Sample Size: {moprobo_data["metrics"]["sample_size"]} ads')
    print(f'    - Data Quality: {moprobo_data["metrics"]["data_quality"]:.0%}')

    print(f'\n  Top Performing Patterns:')
    for i, pattern in enumerate(moprobo_data['top_patterns'], 1):
        features = pattern['features']
        lift = pattern['roas_lift_multiple']
        print(f'    {i}. {features} → {lift}x ROAS lift')

    orchestrator = AgentOrchestrator()

    # Define moprobo-specific evolution rounds
    rounds = [
        {
            'round': 1,
            'objective': 'discover_high_lift_patterns',
            'domain': 'home_goods',  # Moprobo's vertical
            'context': {
                'issue': f'Current top pattern has {moprobo_data["top_patterns"][0]["roas_lift_multiple"]}x lift, need to find higher',
                'current_metrics': {
                    'avg_roas': moprobo_data['metrics']['avg_roas'],
                    'top_pattern_lift': moprobo_data['top_patterns'][0]['roas_lift_multiple'],
                    'sample_size': moprobo_data['metrics']['sample_size'],
                },
                'customer': 'moprobo',
                'vertical': 'home_goods',
            },
        },
        {
            'round': 2,
            'objective': 'optimize_vertical_performance',
            'domain': 'home_goods',
            'context': {
                'issue': f'Home goods vertical ROAS {moprobo_data["metrics"]["avg_roas"]:.2f}x, can improve with better features',
                'current_metrics': {
                    'vertical_roas': moprobo_data['metrics']['avg_roas'],
                    'top_quartile_roas': moprobo_data['metrics']['top_quartile_roas'],
                },
                'customer': 'moprobo',
                'target_roas': 35.0,  # Target 35x ROAS
            },
        },
        {
            'round': 3,
            'objective': 'reduce_false_positive_patterns',
            'domain': None,
            'context': {
                'issue': f'Data quality {moprobo_data["metrics"]["data_quality"]:.0%}, some patterns may be false positives',
                'current_metrics': {
                    'data_quality': moprobo_data['metrics']['data_quality'],
                    'roas_range': moprobo_data['metrics']['roas_range'],
                },
                'customer': 'moprobo',
            },
        },
        {
            'round': 4,
            'objective': 'improve_visual_feature_extraction',
            'domain': None,
            'context': {
                'issue': 'Visual creative features not available in performance dataset (fallback used)',
                'current_metrics': {
                    'extraction_coverage': 0.0,  # Currently using fallback
                    'data_quality': moprobo_data['metrics']['data_quality'],
                },
                'customer': 'moprobo',
                'target_coverage': 0.90,
            },
        },
        {
            'round': 5,
            'objective': 'increase_winner_precision',
            'domain': None,
            'context': {
                'issue': f'Top quartile has {moprobo_data["metrics"]["top_quartile_size"]} ads, need better precision',
                'current_metrics': {
                    'top_quartile_size': moprobo_data['metrics']['top_quartile_size'],
                    'winner_precision': 0.68,  # Estimated from marble prevalence
                },
                'customer': 'moprobo',
                'target_precision': 0.85,
            },
        },
    ]

    results = []

    print('\n' + '='*80)
    print(f'RUNNING {len(rounds)} EVOLUTION CYCLES ON MOPROBO DATA')
    print('='*80)

    for round_config in rounds:
        round_num = round_config['round']
        objective = round_config['objective']
        domain = round_config['domain']
        context = round_config['context']

        print(f'\n[Round {round_num}/{len(rounds)}] {objective}' + (f' ({domain})' if domain else ''))
        print(f'  Customer: moprobo | Product: {moprobo_data["product"]}')
        print(f'  Issue: {context["issue"]}')

        # Run evolution loop
        cycle = orchestrator.run_evolution_loop(
            objective=objective,
            domain=domain,
            context=context,
            max_iterations=2,
        )

        # Extract result
        result = {
            'round': round_num,
            'objective': objective,
            'domain': domain,
            'decision': cycle.final_decision,
            'duration': cycle.duration_seconds,
        }

        # Get lift score from validation phase
        for phase_name, phase_data in cycle.phases.items():
            if 'validation' in phase_name and isinstance(phase_data, dict):
                if 'lift_score' in phase_data:
                    result['lift'] = phase_data['lift_score']
                break

        results.append(result)

        # Display result
        status_icon = '✅' if result['decision'] == 'PASS' else '❌'
        lift_info = f" | Lift: +{result.get('lift', 0):.1f}%" if 'lift' in result else ""
        print(f'  Result: {status_icon} {result["decision"]}{lift_info} ({result["duration"]:.1f}s)')

    # Summary statistics
    print('\n' + '='*80)
    print('MOPROBO SIMULATION SUMMARY')
    print('='*80)

    total_passed = sum(1 for r in results if r['decision'] == 'PASS')
    total_failed = sum(1 for r in results if r['decision'] == 'FAIL')
    avg_duration = sum(r['duration'] for r in results) / len(results)

    print(f'\n📈 RESULTS:')
    print(f'  Total Rounds: {len(results)}')
    print(f'  ✅ Passed: {total_passed} ({total_passed/len(results)*100:.0f}%)')
    print(f'  ❌ Failed: {total_failed} ({total_failed/len(results)*100:.0f}%)')
    print(f'  ⏱️  Average Duration: {avg_duration:.2f}s')

    # Objective breakdown
    print(f'\n📋 ROUND-BY-ROUND BREAKDOWN:')
    print('-'*80)
    for r in results:
        status_icon = '✅' if r['decision'] == 'PASS' else '❌'
        domain_info = f" ({r['domain']})" if r['domain'] else ""
        lift_info = f" | Lift: +{r.get('lift', 0):.1f}%" if 'lift' in r else ""
        print(f"{status_icon} Round {r['round']}: {r['objective']}{domain_info}{lift_info}")

    # Memory Agent stats
    memory_stats = orchestrator.memory_agent.get_statistics()
    print(f'\n🧠 MEMORY AGENT:')
    print(f'  Total Experiments: {memory_stats["total_experiments"]}')
    print(f'  Patterns Stored: {len(orchestrator.memory_agent.experiments)}')

    # Moprobo-specific recommendations
    print(f'\n💡 MOPROBO-SPECIFIC INSIGHTS:')
    print(f'  • Current best pattern: {moprobo_data["top_patterns"][0]["features"]}')
    print(f'    → {moprobo_data["top_patterns"][0]["roas_lift_multiple"]}x ROAS lift')
    print(f'  • Sample size: {moprobo_data["metrics"]["sample_size"]} ads (good statistical power)')
    print(f'  • Data quality: {moprobo_data["metrics"]["data_quality"]:.0%} (needs visual feature extraction)')
    print(f'  • Opportunity: Improve visual feature extraction for better patterns')

    print('\n' + '='*80)
    print('✅ MOPROBO EVOLUTION SIMULATION COMPLETE')
    print('='*80)
    print('\nNext Steps for Moprobo:')
    print('  1. Implement visual feature extraction for all ad creatives')
    print('  2. Re-run pattern mining with complete visual data')
    print('  3. Test new patterns in A/B campaigns')
    print('  4. Monitor ROAS improvement from pattern-based recommendations')
    print('='*80 + '\n')

if __name__ == '__main__':
    main()
