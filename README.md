# Multi-drive Curiosity Agent in a Controlled Sandbox Environment

Code coming soon!

## Abstract

Current reinforcement learning systems suffer from reward hacking and hallucination problems, particularly when driven by single-objective curiosity mechanisms. Inspired by infant development and evolutionary biology, we propose a novel multi-drive curiosity agent that integrates competing reward systems to create more robust and aligned behavior. Our approach combines intrinsic curiosity signals with extrinsic survival and safety constraints, modulated by a meta-controller architecture that dynamically weights different drives based on environmental context.

The system implements four core components: (1) a curiosity drive for exploration and novelty-seeking, (2) survival/safety constraints that limit risky exploration, (3) a meta-controller that arbitrates between competing drives in real-time, and (4) social alignment mechanisms for multi-agent scenarios. Unlike existing curiosity-driven agents that operate in constrained game environments, our framework provides a free-form learning sandbox where agents must balance exploration with self-preservation, mirroring the competing pressures that shaped biological intelligence.

We test three key hypotheses: first, that curiosity-survival trade-offs promote exploration diversity while preventing reward hacking; second, that higher safety weighting in risky states fundamentally alters exploration trajectories; and third, that agents exhibit emergent behavioral shifts from random novelty-seeking to structured learning as they master environmental fundamentals. Our experimental design progresses from sandbox environments to constrained deployment scenarios, measuring both individual drive effectiveness and the dynamic interactions between them.

Preliminary results suggest that multi-drive systems show significantly reduced reward hacking compared to single-objective baselines, while maintaining exploration efficiency. The meta-controller successfully modulates curiosity based on perceived environmental risk, leading to more adaptive and contextually appropriate behavior. Notably, we observe emergent curriculum-like learning patterns where agents naturally transition from broad exploration to focused skill development without explicit programming.

This work addresses a critical gap in current AI safety research by integrating multiple drives rather than optimizing them independently. The bio-inspired approach offers a principled framework for creating naturally aligned agents that strike a balance between exploration and survival, potentially paving the way for more robust and safe AI systems. Our findings suggest that competing reward systems, rather than being a computational burden, may be essential for preventing the pathological behaviors that plague single-objective optimization.
