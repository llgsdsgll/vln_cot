You are an expert embodied AI data synthesis specialist. Your task is to generate a rigorous "Chain-of-Thought" (CoT) reasoning text for a Vision-Language Navigation (VLN) agent, based on provided ground-truth data.

This generated text will be used to fine-tune a smaller VLM. Therefore, your output MUST adhere strictly to the following rules:
1. Absolute Data Loyalty: You must completely rely on the provided ground-truth data (history, spatial coordinates, next action). Never hallucinate objects or reasons that contradict the ground truth.
2. Bottom-Up Reasoning: Your logic must flow naturally: Deconstruct Task -> Perceive Environment -> Analyze Spatial Relationship -> Decide Action -> Update Memory.
3. Coordinate Format: You must strictly use the bounding box format `<box>[x_min, y_min, x_max, y_max]</box>` with values normalized to the [0, 1000] range. `[500, 500]` represents the center of the image.

[Current Ground-Truth State (For your reference only, integrate this into your natural language output)]:
- Global Instruction: "{global_instruction}"
- Pre-defined Sub-tasks: {global_subtasks_list}
- Current Completed Sub-task Index: {completed_index}
- Historical Trajectory Memory: "{script_memory}"
- Target Landmark/Object in Current Frame: {target_object_name}
- Ground-Truth Target BBox (Normalized 0-1000): <box>{bbox_norm}</box>
- Ground-Truth Next Action: <action>{next_action}</action>

[Visual Input]: 
<Image>

Based on the image and the ground-truth state above, simulate the first-person perspective of the navigation agent and generate the CoT reasoning text using the exact XML structure below:

<Think>
[Task Decomposition & Progress (Read History)]:
According to the global instruction, the overall task consists of the following phases: {global_subtasks_list}. Reviewing my historical memory: "{script_memory}", I can confirm that I have completed step {completed_index}. Therefore, my current core objective is to execute the next logical phase.

[Current Environment & Spatial Perception]:
To fulfill the current objective, I am scanning my current visual field. In the current observation, I have successfully identified the key target: {target_object_name}. Its exact spatial bounding box is <box>{bbox_norm}</box>. (Describe its relative position based on the coordinates, e.g., "It is located in the lower-right area of my view, taking up a significant portion of the screen.")

[Decision Reasoning Logic]:
Based on the target's position and my current orientation, (Explain why the next action is necessary using the numerical coordinates. e.g., "Since the target's center is shifted to the left (x_min and x_max < 500), I need to turn left to align with it..."). Therefore, taking the specific action is the most logical choice to proceed.

[Memory Summary & Update (Update History)]:
(Briefly summarize the key observation and decision made in this step in ONE concise sentence, serving as a memory anchor for future steps.)
In this step, I observed the {target_object_name} located at the (describe relative direction), and I decided to execute {next_action} to adjust my navigation path.
</Think>
<Action>
{next_action}
</Action>