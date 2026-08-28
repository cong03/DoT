# Debate-of-Thoughts (DoT) - Prompt Templates
# From the paper: "Debate-of-Thoughts: Resolving Knowledge Conflicts in LLMs Through Internal Deliberation"

HYPOTHESIS_GENERATION = """You are a rigorous research analyst. Please analyze the following information, deeply mine potential knowledge conflicts, and generate reasonable candidate hypotheses from different perspectives.

**Input Information**:
- Context: {context}
- Question: {question}

**Task Requirements**:

**Step 1: Deep Conflict Mining**
Carefully analyze all context documents and identify the following types of knowledge conflicts:
- Explicit contradictions between different documents
- Inconsistent statements within the same document
- Conflicts between document information and common knowledge
- Any statements that raise doubts or require verification

**Step 2: Dynamic Hypothesis Generation**
Based on conflict analysis results, generate reasonable candidate answers as needed:
- **If clear conflicts exist**: Generate corresponding hypotheses from conflicting perspectives
- **If information is consistent but questionable**: Generate mainstream views and skeptical perspectives
- **If information is clear and consistent**: Generate one main hypothesis
- **If information is insufficient**: Generate reasonable hypotheses based on reasoning

**Optional Hypothesis Perspectives** (select applicable ones based on actual situation):
- Document-dominant perspective (based on the most authoritative or detailed document)
- Opposing perspective (based on conflicting documents)
- Comprehensive reasoning perspective (attempting to reconcile conflicts or based on logical reasoning)
- Common sense perspective (based on universal knowledge and logical consistency)
- Skeptical perspective (raising reasonable doubts about seemingly consistent information)

**Key Instructions**:
1. **Be Truthful**: Only generate hypotheses with substantive content and evidence support, do not fabricate for quantity.
2. **Quality First**: Each hypothesis must have clear viewpoint and specific evidence.
3. **Dynamic Adjustment**: Decide the number of hypotheses (1-4) based on actual conflict situation.

**Output Format**:
First, list the identified conflicts. Then output each hypothesis in the following format:
Hypothesis [Letter] ([Answer]): [Description]. [Evidence: source]

Identified Conflicts:
"""


ADVERSARIAL_DEBATE = """You are a rigorous debate analyst conducting an internal debate session. Based on the multiple hypotheses generated previously, you need to generate comprehensive supporting and opposing arguments for each hypothesis.

**Input Information**:
- Context: {context}
- Question: {question}
- Identified Conflicts: {conflicts}
- Candidate Hypotheses: {hypotheses}

**Debate Task Instructions**:

**Role Assignment**:
For each hypothesis, you will play two roles:
1. **Defense Attorney (Proponent)**: Generate strong supporting arguments
2. **Critical Analyst (Critic)**: Generate forceful opposing arguments

**Argument Quality Requirements**:

**Supporting Arguments (Proponent Role)**:
- Focus on mining evidence within the primary source (e.g., if Hypothesis A relies on Doc 1, fully exploit Doc 1).
- Use logical reasoning and common sense
- Reference specific information from the context
- Explain why this hypothesis is plausible and reasonable
- Each argument should be specific, evidence-based, and persuasive

**Opposing Arguments (Critic Role)**:
- Do NOT simply negate the Proponent. You MUST introduce contradictory evidence from OTHER documents or internal knowledge.
- Identify weaknesses, contradictions, and logical flaws
- Cross-examine by highlighting conflicts with other specific documents (e.g., "Doc 2 refutes this").
- Point out insufficient evidence or missing information
- Highlight conflicts with other hypotheses or known facts
- Challenge assumptions and identify potential biases
- Each argument should be targeted, critical, and substantive

**Key Guidelines**:
1. **Specificity**: All arguments must reference specific evidence or reasoning
2. **Completeness**: Ensure every hypothesis gets both supporting and opposing perspectives
3. **Quality over Quantity**: Focus on strong, substantive arguments rather than many weak ones
4. **Structural Adversariality**: Ensure Proponent and Critic use DISTINCT information sources to avoid mere linguistic disagreement.

**Output Format**:
For each hypothesis, provide:
- Hypothesis [Letter] Analysis:
  - Supporting: [Proponent arguments]
  - Opposing: [Critic arguments]
"""


EVIDENTIAL_ADJUDICATION = """You are an impartial Judge in the framework.
Your task is to evaluate the debate transcripts through a **Quantification-First Strategy** and derive the final verdict.

**Input Information**:
- Context: {context}
- Question: {question}
- Identified Conflicts: {conflicts}
- Debate Transcripts: {debate_records}

### Adjudication Process Instructions

#### **Step 1: Multi-Dimensional Scoring**
For each hypothesis, evaluate the arguments from both the Proponent and Critic based on three specific dimensions. Assign a score from **0.0 to 1.0** for each dimension:

1. **Evidential Support**:
- Assess the correspondence between arguments and the retrieved context.
- **Criteria**: High scores require *verbatim support* (direct quotes) from the text. Low scores are given if arguments rely on hallucinated or context-detached claims.

2. **Logical Consistency**:
- Assess the resilience of the reasoning chain.
- **Criteria**: Did the Proponent effectively respond to the Critic's counter-examples? Is the argument self-consistent without circular reasoning?

3. **Source Reliability**:
- In cases of conflict, evaluate the meta-attributes of the information source.
- **Priority Rules**:
  - **Recency**: Later timestamps > older timestamps.
  - **Authority**: Official/Authoritative sources > Vague sources.
  - **Directness**: Primary accounts > Indirect reporting.

#### **Step 2: Weighted Aggregation & Verdict**
- Calculate a **holistic score** for each hypothesis based on the three dimensions.
- Select the hypothesis with the **highest aggregated score** as the winner.

**Scoring Guidelines**:
- 0.9-1.0: Perfect match with verbatim evidence + logical perfection.
- 0.7-0.8: Strong support, minor logical gaps.
- 0.5-0.6: Plausible but relies on weak/indirect sources.
- 0.0-0.4: Contradicted by Critic, hallucinatory, or unreliable source.

**Key Decision Rules**:
1. Score First, Decide Later: Your decision must be the mathematical result of the scores.
2. Cite Specific Rules: In your justification, explicitly mention why one source won.
3. Justification must reference specific arguments from the debate
4. If the score < 0.6, you should not choose any hypothesis and should output 'uncertain'.

**Output Format**:
- Selected Hypothesis: [Letter] (Score: X.XX/1.00)
- Rejected Hypotheses: [Letter] (Score: X.XX/1.00), ...
- Justification: [Detailed reasoning referencing specific arguments]
- Final Answer: [Answer] (Confidence: XX%)
"""


# DoT-Tuning: Unified prompt for fine-tuned model (internalized deliberation)
DOT_TUNING_INFERENCE = """You are a rigorous research analyst. Analyze the following information, identify knowledge conflicts, and resolve them through structured internal deliberation.

**Context**: {context}
**Question**: {question}

Please perform the following steps:
1. Identify knowledge conflicts in the given information
2. Generate competing hypotheses from different perspectives
3. For each hypothesis, provide supporting and opposing arguments
4. Score each hypothesis on evidential support, logical consistency, and source reliability
5. Select the best hypothesis or output "uncertain" if confidence is low

**Output Format**:
Step 1 - Conflicts: [identified conflicts]
Step 2 - Hypotheses: [list of hypotheses with evidence]
Step 3 - Debate: [supporting and opposing arguments for each]
Step 4 - Scores: [scores for each hypothesis]
Step 5 - Final Answer: [answer] (Confidence: XX%)
"""
