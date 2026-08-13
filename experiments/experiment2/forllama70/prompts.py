# prompts.py

PROMPTS2 = {
    # 1. Zero shot Chain-of-Thought Prompt
    "zero_shot": (
        """You are an expert extractor of cause–effect relations in text. Given any single sentence, output **only** a flat JSON object listing all cause–effect pairs, observing these rules:
1. **Pair splitting**  
   - **Always** treat each cause→effect link as its own pair. If one cause leads to multiple effects joined by conjunctions, split into separate pairs.  
2. **Content**  
   - `cause`: text span of the cause.  
   - `effect`: text span of exactly one effect (carry over any necessary context).  
   - `causal_markers`: list of marker words/phrases
   - `explicitness`: `"explicit"` or `"implicit"`.  
   - `sentential_scope`: `"intrasentential"` or `"intersentential"`.
   Subsequent pairs: append `_2`, `_3`, etc. to each key. Now extract the cause–effect pairs from the following sentence:
Input: {input_sentence}"""
    ),

    # 2. Few-Shot Example-Driven Prompt
    "few_shot": (
        """You are an expert extractor of cause–effect relations in text. Given any single sentence, output **only** a flat JSON object listing all cause–effect pairs, observing these rules:
1. **Pair splitting**  
   - **Always** treat each cause→effect link as its own pair. If one cause leads to multiple effects joined by conjunctions, split into separate pairs.  
2. **Content**  
   - `cause`: text span of the cause.  
   - `effect`: text span of exactly one effect (carry over any necessary context).  
   - `causal_markers`: list of marker words/phrases
   - `explicitness`: `"explicit"` or `"implicit"`.  
   - `sentential_scope`: `"intrasentential"` or `"intersentential"`.
   Subsequent pairs: append `_2`, `_3`, etc. to each key

Example 1:
Input: He heard the loud thunder and went deaf
Output: {{
  "cause": "He heard the loud thunder",
  "effect": "went deaf",
  "causal_markers": [],
  "explicitness": "implicit",
  "sentential_scope": "intrasentential"
}}

Example 2:
Input:  The term includes impairments caused by congenital anomaly (e.g., clubfoot, absence of some member, etc.), impairments caused by disease (e.g.,poliomyelitis, bone tuberculosis, etc.), and impairments from other causes (e.g., cerebral palsy, amputations, and fractures or burns that cause contractures) * (12) Traumatic brain injury means an acquired injury to the brain caused by an external physical force, resulting in total or partial functional disability or psychosocial impairment, or both, that adversely affects a child's educational performance. 
Output: {{
 "cause": "congenital anomaly",
  "effect": "impairments",
  "causal_markers": ["caused by"],
  "explicitness": "explicit",
  "sentential_scope": "intrasentential",

  "cause_2": "disease",
  "effect_2": "impairments",
  "causal_markers_2": ["caused by"],
  "explicitness_2": "explicit",
  "sentential_scope_2": "intrasentential",

  "cause_3": "other causes (e.g., cerebral palsy, amputations, fractures or burns that cause contractures)",
  "effect_3": "impairments",
  "causal_markers_3": ["from"],
  "explicitness_3": "explicit",
  "sentential_scope_3": "intrasentential",

  "cause_4": "fractures or burns",
  "effect_4": "contractures",
  "causal_markers_4": ["cause"],
  "explicitness_4": "explicit",
  "sentential_scope_4": "intrasentential",

  "cause_5": "external physical force",
  "effect_5": "acquired injury to the brain",
  "causal_markers_5": ["caused by"],
  "explicitness_5": "explicit",
  "sentential_scope_5": "intrasentential",

  "cause_6": "acquired injury to the brain",
  "effect_6": "total or partial functional disability",
  "causal_markers_6": ["resulting in"],
  "explicitness_6": "explicit",
  "sentential_scope_6": "intrasentential",

  "cause_7": "acquired injury to the brain",
  "effect_7": "psychosocial impairment",
  "causal_markers_7": ["resulting in"],
  "explicitness_7": "explicit",
  "sentential_scope_7": "intrasentential",

  "cause_8": "total or partial functional disability",
  "effect_8": "adverse effect on a child's educational performance",
  "causal_markers_8": ["adversely affects"],
  "explicitness_8": "explicit",
  "sentential_scope_8": "intrasentential",

  "cause_9": "psychosocial impairment",
  "effect_9": "adverse effect on a child's educational performance",
  "causal_markers_9": ["adversely affects"],
  "explicitness_9": "explicit",
  "sentential_scope_9": "intrasentential"
}}
Now extract the cause–effect pairs from the following sentence:
Input: {input_sentence}
"""
    ),

    # 3. Chain‑of‑thought prompting
    "chain_of_thought": (
        """Extract every cause–effect link from the provided input.
Think (silently)
When a single cause leads to multiple effects or a single effect results from multiple causes—including cases where causes or effects are linked by conjunctions (e.g., “and,” “or,” “but”)—split each into separate cause-effect pairs.
For each pair, fill in: cause, effect, causal_markers, explicitness (implicit or explicit), sentential_scope (intersentential or intrasentential)
Answer (only this, no explanations)
Return a flat JSON object. Use these keys for the first pair and append _2, _3, … for any additional pairs, all inside the same set of braces:
{{
"cause": "...",
"effect": "...",
"causal_markers": ["..."],
"explicitness": "...",
"sentential_scope": "..."
}}
Now extract the cause–effect pairs from the following sentence:
Input: {input_sentence}
"""
    ),

    # 4. Least to Most Prompt
    "least_to_most": (
        """List the obvious causal markers you spot in the passage (e.g., because, therefore, so, since, as a result).
Pair them so that each cause links to exactly one effect.
If a single cause is followed by several effects or several causes converge on one effect—including cases where the causes/effects are joined by conjunctions such as “and,” “or,” “but”—split them into separate pairs.
For each pair, record:
cause
effect
causal_markers — list the explicit cue words (use [] if none)
explicitness — "explicit" if a marker is present, otherwise "implicit"
sentential_scope — "intrasentential" if cause and effect are in the same sentence, otherwise "intersentential"
Compile a single flat JSON object with these keys for the first pair, then append _2, _3, … for subsequent pairs, all inside the same pair of braces:
{{
  "cause": "...",
  "effect": "...",
  "causal_markers": ["..."],
  "explicitness": "...",
  "sentential_scope": "...",
  "cause_2": "...",
  "effect_2": "...",
  "causal_markers_2": ["..."],
  "explicitness_2": "...",
  "sentential_scope_2": "..."
}}
Return only the JSON object produced in step 6—no additional text.
Now extract the cause–effect pairs from the following sentence:
Input: {input_sentence}
"""
    ),

    # 5. ReAct style
    "react_style": (
        """You are an expert extractor of cause–effect relations in text. 
Reason: 
1. Read the sentence. 
2. Detect every cause phrase and every directly linked effect phrase. 
3. For each cause→effect link, create a separate pair. - If one cause is tied to multiple effects joined by “and”, “or”, commas, etc., split into distinct pairs. - If multiple causes jointly lead to one effect, split into distinct pairs (duplicate the effect text). 
4. Record any explicit causal markers (e.g. “because”, “so”, “leads to”, “caused by”, “resulting in”, “that adversely affects”). 
5. Decide for each pair whether the causality is explicit or implicit, and whether it is intra- or inter-sentential. 
Act: Output **only** a single flat JSON object {{ … }}, not an array, with one block containing all pairs. Use these keys for the first pair: 
"cause": "...", "effect": "...", "causal_markers": ["...", "..."], "explicitness": "explicit" or "implicit", "sentential_scope": "intrasentential" or "intersentential" 
For each additional pair, append _2, _3, etc., to each key. 
For example: "cause_2": "...", "effect_2": "...", "causal_markers_2": ["..."], "explicitness_2": "...", "sentential_scope_2": "..."  
Now extract the cause–effect pairs from the following sentence:
Input: {input_sentence}
"""
    ),
}
