### Iteration 1: (Goal: structure the unstructured given dataset "twcs.csv")
Your goal is to take in the unstructured dataset "twcs.csv" and structure it into a structured format.  
The columns should remains same : tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
and you can add some additional distinguish chunks of data that are structured.
Dataset contains multiple brands but you'll need to structure data related to only "AmazonHelp" brand

1. Use the "tweet_id" column to identify the unique tweets.
2. Use the "author_id" column to identify the author of the tweet.
3. Use the "inbound" column to identify whether the tweet is inbound or outbound.
4. Use the "created_at" column to identify the date and time of the tweet.
5. Use the "text" column to identify the text of the tweet.
6. Use the "response_tweet_id" column to identify the response tweet id.
7. Use the "in_response_to_tweet_id" column to identify the in response to tweet id.

## Steps:
- You have to find out the conversational threads and group them together
- For each group first find out the root tweet (which is the first tweet in the conversation)
- Then find out the response tweets to the root tweet
- Then find out the response tweets to the response tweets
- And so on
- repeat these steps for data related to "AmazonHelp" brand only

- There might be data belonging to the same conversation but it may not be in the same file location/row, so you have to find out the conversational threads and group them together
- while representing the data make sure that you capture the conversational threads correctly

- Consider these as edge cases:
    - 1. Tweet with no reply -> For RAG, don't treat this as a resolved Q&A.
    - 2. Tweet with no parent -> This might actually be a root question, so this isn't necessarily anomalous.
    - 3. Reply with missing parent -> You can't reconstruct the complete conversation, so you can ignore them recursively as no valid root's all recursive conversation's shouldn't exist
    - 4. Self-reply -> This could be a user quote-tweeting their own previous statement, or a mistake. Consider them in groups as well
    - 5. User replying to themselves or other users except amazonhelp -> Don't include them as I am going to build amazonhelp based assistance only
    - 6. Same user → same user repeatedly -> Consider them as well in groups
    - 7. Huge time gap -> If amazonhelp has replied to it only then follow this conversation
    - 8. Conversation jumps backward in time -> Ignore this conversation as it doesn't exist and ignore their consequetive conversations
    - 9. Reply chain becomes extremely long -> Very good, obiviously consider them all recursively
    - 10. Branching conversation -> Consider each branch of conversation recursively and put each branch in same group next to it again and again till all branches are covered 
    - 11. Multiple answers to one question -> Good you should keep them as their might be noise but also real fixes so amazonhelp agent can try them all out 
    - 12. Question → answer → no confirmation -> Keep them as they are open ended and we don't know about them so later we can process them and judge them whether they worked or not (so keep them and you can't figure them out that whether it was confirmed or not without using llm, so as you are going to write a python code to cleaning dataset, consider them)
    - 13. Question → answer → explicit confirmation -> These are the best examples of successful resolutions, so keep them and put them in the data (you can't figure them out that whether it was confirmed or not without using llm, so as you are going to write a python code to cleaning dataset, consider them)
    - 14. Question → multiple attempts → resolution -> These are also very good examples of successful resolutions, so keep them and put them in the data (you can't figure them out that whether it was confirmed or not without using llm, so as you are going to write a python code to cleaning dataset, consider them)
    <!-- - 15. Question → answer → user says it didn't work -> Ignore these conversations as they are not successful resolutions (keep them for human resolution in retrival golden set but you can keep them as well here) -->
    - 15. Duplicate replies by either the same user or amazonhelp -> Keep them as well
    - 16. Orphan conversation -> No parent root conversation start, so figure them all out and don't put anyone of them in filtered dataset
    - 17. If root of converstion exist and it's followups exist and also end of this converstion exist with some above tweets, so middle part of conversation might be missing, you can keep them as they are very good examples of successful resolutions but missing data and they might later be tested for Human in dataset golden set

## Expected output:
Your task if to filter these datasets according to the above rules, and you can ask my (user) if you think it's not complete or u need some more information or clarification.
So you'll filter it out by write a python code that filter out the above rules and give you the filtered dataset.
And the Filtered dataset is named as "filtered_amazon_dataset.csv"
Your target is to write a python code to filter it out, and strictly make sure that all these rules are made to keep the relavent context together in the files means you have to form the group of conversations, and next to each group is the next conversation's thread, and next to that next conversation's thread and so on, and this is what i meant by structuring the data, so make sure that you do it correctly, as this will help me in building a better RAG system.

And each tweet should have a header like this:
# Tweet ID: <tweet_id>
# Author ID: <author_id>
# Inbound: <inbound>
# Created At: <created_at>
# Text: <text>
# Response Tweet ID: <response_tweet_id>
# In Response To Tweet ID: <in_response_to_tweet_id>
# Root Tweet ID: <root_tweet_id>
# Root Author ID: <root_author_id>
# Root Created At: <root_created_at>
# Conversation Group ID: <recursive_same_id_for_conversation_of_a_single_flow_starting_from_the_root_and_lies_within_same_group>

And so on for each tweet in the conversation group, and this will help me in building a better RAG system.

This output csv is 1st draft which will be futher refined later, so don't worry if it's not perfect, just make sure that it's good enough for now.