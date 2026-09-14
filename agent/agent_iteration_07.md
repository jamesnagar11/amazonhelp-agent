## Iteration 07 : (Goal: To reject the for user's query which are not related to amazon or its products or based on some personal agenda or if it's gibberish. And shouldn't escalate to human_node directly. Instead reject them straight away with a reason, so the graph state is also modified according to it. All the necessary information I am providing below. Also dont touch or change any other functionality of the current graph or its nodes at all. Keep everything as it is. )

### First we adjust the get_intent node

- The get_intent node currently expects dict of "intent", "intent_score", "detected_language" but now it's role is also to also provide and include "should_reject" and also "reject_reason" (only when should_reject is true)
- The structed output should esure these additional attributes of the output from this node and should be reflected correctly in the graph state.
- "reject_reason" will be provided by the llm based on the user input query.
- "reject_reason" will be left empty or null if should_reject is false.
- Also make sure llm is correctly rejecting the requests.

- You should also modify the current get_intent prompt as well to do the job properly. But please make sure it shouldn't affect the prompt quality of current level intent detection. So, the prompt should be engineered properly for it as well as should be 1st priority to not affect the llm output quality for intent detection but also add another value to the output the llm going to provide which is the correct reject reason.

## Adding conditional edge correctly
- Make a conditional edge from get_intent based on the value of "should_reject" to directly END of the graph, instead of human node.

## Graph state
- Graph state should be updated with 2 more properties/attributes named "reject" with boolan type and "reject_reason" with string type or null and should be null if should_reject is false. And if should_reject is true then reject_reason should be the reason for rejection provided by the llm as string.

- This change should be reflected properly in the graph state, without affecting any of the current working of the graph or it's components.

Make sure it doesn't affect any of the other nodes's input or output at all. And also make sure it correctly updates the graph state and the structed output for get_intent and prompt for get_intent.

If you have any ambiguity in this task then you can ask me, don't assume or decide by your own, and also don't affect or change any of the current appliation other that these states rules. Keep everything as it is.

- If at any point you are not able to understand any of the instructions or points, then you can ask me, don't assume or decide by your own, and also don't affect or change any of the current appliation other that these states rules. Keep everything as it is.

##

Good Luck!