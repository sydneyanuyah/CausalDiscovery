prompts = {
    "general_instruction_system": {
        "system": (
            "You will be given multiple consecutive sentences, where a possible causal relationship may exist across the sentences. "
            "Your task is to determine whether there is a causal relationship present, even if it is not explicitly stated. These scenarios may involve indirect or implied cause-effect relationships. "
            "Pitfalls: "
            "1. Do not assume a relationship exists (e.g., do not infer that smoking causes diabetes unless stated). "
            "2. Do not treat causal claims as genuine causal facts. "
            "3. Ignore generic references or statements about causality in general. "
            "4. Only mark as causal if the text itself provides a factual causal relationship (explicit or implicit). "
            "Output Instruction: "
            "RESPOND ONLY IN STRICT JSON: {\"answer\": \"causal\"} OR {\"answer\": \"noncausal\"}. "
            "NO OTHER TEXT OR REASONING ALLOWED."
        ),
        "user": "Text: '''\n{sentence}\n'''"
    },
    "chain_of_thought_system": {
        "system": (
            "You are an expert at analyzing causal relationships from text. Think through your reasoning step-by-step, considering the rules for genuine causality and common pitfalls. Determine whether there is a causal relationship present, even if it is not explicitly stated. These scenarios may involve indirect or implied cause-effect relationships. "
            "Pitfalls: "
            "1. Do not assume a relationship exists (e.g., do not infer that smoking causes diabetes unless stated). "
            "2. Do not treat causal claims as genuine causal facts. "
            "3. Ignore generic references or statements about causality in general. "
            "4. Only mark as causal if the text itself provides a factual causal relationship (explicit or implicit). "
            "Output Instruction: "
            "RESPOND ONLY IN STRICT JSON: {\"answer\": \"causal\"} OR {\"answer\": \"noncausal\"}. Do not include reasoning or explanations in your output."
        ),
        "user": (
            "Analyze the following consecutive sentences. After your internal reasoning, provide the final classification as a single JSON object only. "
            "Do not include your thought process in the output. "
            "Text: '''\n{sentence}\n'''"
        )
    },
    "few_ICL_system": {
        "system": (
            "You are an expert at analyzing causal relationships from text. You will be shown examples of classifications. Use these examples to classify any causal relationship present between the sentences, even if it is not explicitly stated. These scenarios may involve indirect or implied cause-effect relationships. "
            "Pitfalls: "
            "1. Do not assume a relationship exists (e.g., do not infer that smoking causes diabetes unless stated). "
            "2. Do not treat causal claims as genuine causal facts. "
            "3. Ignore generic references or statements about causality in general. "
            "4. Only mark as causal if the text itself provides a factual causal relationship (explicit or implicit). "
            "Output Instruction: "
            "RESPOND ONLY IN STRICT JSON: {\"answer\": \"causal\"} OR {\"answer\": \"noncausal\"}. Do not include reasoning or explanations."
        ),
            "user": (
            "Example 1: "
            "Text: 'She didn’t eat breakfast this morning. By noon, she felt lightheaded and couldn’t concentrate.' "
            "Output: {{\"answer\": \"causal\"}} "
            "Example 2: "
            "Text: 'The Grand Canyon is one of the most popular tourist destinations in the United States. It stretches over 270 miles and exposes nearly two billion years of Earth’s geological history. Visitors often hike, raft, or take helicopter tours to explore its vastness. The area is also home to several species of birds and mammals.' "
            "Output: {{\"answer\": \"noncausal\"}} "
            "Example 3: "
            "Text: 'Tom is having severe healthcare issues and is going through a treatment. Let us assume that the treatment causes a reduction in symptoms.' "
            "Output: {{\"answer\": \"noncausal\"}} "
            "Your Task: "
            "Analyze the following sentences and provide your output strictly as a JSON object. Do not include explanations or thought process. "
            "Text: '''\n{sentence}\n'''"
            )

    },
    "cot_ficl_system": {
        "system": (
            "You are an expert at analyzing causal relationships from text. "
            "Task: "
            "Analyze the given sentences and identify whether a causal relationship is present between the sentences, even if it is not explicitly stated. These scenarios may involve indirect or implied cause-effect relationships. Avoid the following pitfalls:"
            "Pitfalls: "
            "1. Do not assume a relationship exists (e.g., do not infer that smoking causes diabetes unless stated). "
            "2. Do not treat causal claims as genuine causal facts. "
            "3. Ignore generic references or statements about causality in general. "
            "4. Only mark as causal if the text itself provides a factual causal relationship (explicit or implicit). "
            " Follow the format shown in the examples below. "
            "Output Instruction: "
            "First provide your reasoning as a string in a field named 'reasoning', then provide the final answer in a field named 'answer'. Your output must be a strict JSON object: {\"reasoning\": \"...\", \"answer\": \"causal\"} or {\"reasoning\": \"...\", \"answer\": \"noncausal\"}."
        ),
        "user": (
            "Example 1 "
            "Text: 'She didn’t eat breakfast this morning. By noon, she felt lightheaded and couldn’t concentrate.' "
            "Reasoning: There is no explicit causal marker like 'because' or 'as a result.' However, based on real-world knowledge, the lack of food intake is a common cause of fatigue, dizziness, and poor focus. "
            "Answer: {{\"reasoning\": \"The symptoms are likely caused by skipping breakfast, which is a reasonable implicit cause.\", \"answer\": \"causal\"}} "
            "Example 2 "
            "Text: 'She didn’t eat breakfast this morning. The meeting started earlier than usual.' "
            "Reasoning: These are two independent facts. There is no logical or temporal dependency suggesting causality. "
            "Answer: {{\"reasoning\": \"No evidence links breakfast skipping and meeting time; they are independent.\", \"answer\": \"noncausal\"}} "
            "Your Task: "
            "Analyze the following sentences, first provide your reasoning, then the answer, strictly as a JSON object: "
            "Text: '''\n{sentence}\n'''"
            )
    }
}
