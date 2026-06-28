# CCI Chunking Strategy Comparison

This report compares the previous CCI vector index with the current structure-aware CCI index using the same retrieval queries.

## Collections
- Previous index: `mental_health_rag`
- Current index: `mental_health_rag_v2`

## Summary
- Previous average top score: `0.8556`
- Current average top score: `0.8506`
- Previous average top chunk size: `125` words
- Current average top chunk size: `193.2` words
- Previous average title diversity in top 5: `2.75`
- Current average title diversity in top 5: `3`

## Recommendation
Use `mental_health_rag_v2` as the production index. The current CCI chunks are bounded, easier for the LLM to use, and avoid sending oversized worksheet-sized passages into generation.

Cosine scores are retrieval similarity signals, not correctness probabilities. The final quality check should combine this report with manual answer review.

## Query-Level Results

### What can help during a panic attack at work?
- Previous top result: `Reassurance Seeking Carers` / `Anxiety` / score `0.8466` / `95` words
- Current top result: `Situational Exposure` / `Panic` / score `0.8317` / `377` words

### How can I stop worrying at night?
- Previous top result: `Postpone your Worry` / `Worry and Rumination` / score `0.8507` / `163` words
- Current top result: `Postpone your Worry` / `Worry and Rumination` / score `0.8481` / `87` words

### What should I do when I keep seeking reassurance?
- Previous top result: `Reducing Reassurance Seeking` / `Anxiety` / score `0.8474` / `168` words
- Current top result: `Reducing Reassurance Seeking` / `Anxiety` / score `0.8577` / `100` words

### How can I improve low self-esteem?
- Previous top result: `What Maintains Low Self-Esteem` / `Self Esteem` / score `0.8632` / `46` words
- Current top result: `Adjusting Negative Core Beliefs` / `Self Esteem` / score `0.8596` / `211` words

### What are practical ways to manage procrastination?
- Previous top result: `Practical Strategies` / `Procrastination` / score `0.8704` / `179` words
- Current top result: `Practical Strategies` / `Procrastination` / score `0.8642` / `396` words

### How can I calm health anxiety?
- Previous top result: `Reassurance Seeking Carers` / `Anxiety` / score `0.8752` / `95` words
- Current top result: `Anxiety and Exercise` / `Anxiety` / score `0.8588` / `224` words

### What can help with social anxiety before meeting people?
- Previous top result: `What can be done about Social Anxiety` / `Social Anxiety` / score `0.854` / `134` words
- Current top result: `What can be done about Social Anxiety` / `Social Anxiety` / score `0.8531` / `87` words

### How do I handle perfectionism when it makes me stuck?
- Previous top result: `What Maintains Perfectionism` / `Perfectionism` / score `0.8369` / `120` words
- Current top result: `What Maintains Perfectionism` / `Perfectionism` / score `0.8318` / `64` words
