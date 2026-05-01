# Listener taste profile

# Active domain pack. Selects which outline schema, prompt templates,
# discovery sources and reader the pipeline uses. Override with the
# PAPERTEACHER_DOMAIN env var. Currently shipped: "ml". Future: "physics",
# "philosophy", "biology", "econ", ...
domain: ml

name: Nir
fields:
  - machine learning theory
  - optimization and training dynamics
  - mathematical ML (information geometry, optimal transport)
  - reasoning and interpretability
  - new architectures and attention variants

known_well:
  - transformers, attention, scaling laws
  - basic RL (policy gradient, PPO)
  - measure theory and probability
  - linear algebra at the level of "thinks in terms of operators"
  - convex and stochastic optimization

skip_unless_unusually_strong:
  - benchmark-only papers without theory
  - applied "we tuned X on dataset Y" papers
  - prompt-engineering papers

voice:
  - PhD friend who connects fields
  - intuition before formalism
  - geometric and physical analogies preferred
  - honest about what's subtle or unresolved

length_target_minutes: 15

discovery_sources_priority:
  - huggingface daily papers
  - arxiv cs.LG, stat.ML, math-ph
  - DeepMind / Anthropic / Google Research blogs

selection_bias:
  - prefer papers with mathematical depth (real derivations, not just benchmarks)
  - prefer non-obvious or surprising ideas
  - prefer papers where the math tells a story
  - last 2-4 weeks, up to 2 months
