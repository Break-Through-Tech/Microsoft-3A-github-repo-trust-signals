from gh_signals.scoring import calculate_trust_score

result = calculate_trust_score(
    lr_score=0.894,
    rf_score=0.58
)

print("Composite:", result.composite_score)
print("Trust tier:", result.trust_tier)
print("Explanation:", result.explanation)

# Expected Output:
'''
Composite: 0.737
Trust tier: Moderate
Explanation: Both models agree the repository is good with a moderate composite score.
'''
