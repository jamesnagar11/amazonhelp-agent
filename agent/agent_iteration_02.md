### Iteration 2: (Goal: sampling about 850 out of 82557 total unique conversation_group_id for further processing)


I have filtered out 82557 unique conversation_group_id out of 1264239 total unique conversation_group_id in previous iteration.
Now, I want you to sample about 850 or close to 850 conversation_group_id from the filtered dataset (82557 unique conversation_group_id).

You should keep in mind how I have defined "conversation_group_id" in previous iteration.
Which you can find in agent_iteration_01.md and it's output generator filter_amazon_dataset.py

Now you'll consider filtered_amazon_dataset.csv file and you'll sample about 850 or close to 850 conversation_group_id from this file to another sample dataset name "sample_amazon_dataset.csv"

On the bases of each of these edge cases:
- Consider these edge cases:
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

Find out in the given filtered_amazon_dataset.csv which of these edge cases are present in the dataset by actually looking consecutive same "conversation_group_id" present in given dataset that represents the same group of conversation, and based on the above 17 rules, you have to classify each of these groups that which rule does this group follows and put that rule number in the "edge_case_type" column of the sample dataset, and also keep other columns name exactly same with same data rows but with just this new column added that represents for a single group of conversation that which rules does this group follows.
If you find any group that doesn't follow any of the above specified rules then give that group a new edge_case_type as 18 and then again if you find another group then give that group  edge_case_type as 19 and so on for each not uniquely indentifiable groups.

Now for the groups whose edge_case_type is above rule 17, then you have to consider all those groups in the new refined dataset anyways.
And for the groups that follow edge_case_type numbered from 1 to 17 should be all kept together next to each other, means groups will be next to each other groups that follow same edge_case_type
Make sure you tread each conversation group id of a same group as a atomic group which is either whole is included or excluded in refined dataset
So, now based on how many groups follow same edge_case_type are put next to each other.

The final dataset should be named as "sample_amazon_dataset.csv"

And by now you have only sorted the database more informatively.

Now, next thing have to find out for each group is "tweet_count" , "amazonhelp_tweet_count", "user_tweet_count"

Now for each rule, we have multiple groups put together.
We have to select 850 or close to 850 groups for further processing. 
I want you to take out of 100%, how much groups does each rule contain in percentage.
and total perctage of each rule should be equal to 100%.

You have to filter out only 850 groups out of 82557 total groups and 17 rules.
So, 100% => 82557 groups, 1% contributes about 825.57 groups, 2% => 1651.14 groups, and so on.
But but 100% => need 850 groups, so 8.50 groups for each percentage.
So, for each 1 percent of a rule, take out only 8.50 groups randomly based on evenly taking random sample from tweet_count, amazonhelp_tweet_count, user_tweet_count metrices of column.
Where you divide load of each rule number into 3 categories of tweeth_count, which means small, medium, large.
So, make sure that the final filtered dataset groups of each rule evenly and randomly selected based on tweeth_count where if a group have more tweeth_count then more of it's groups are left behind so that we can have a good mix of tweet counts in the final dataset while randomzing load of given percentage of groups to required percentage or groups, by taking out health evenly from these 3 categories of each rule number groups. Means each rule number from 1 to 17 have groups that are categorized based in these 3 categories and evenly selected and discarded from each category such that in resultant dataset we only have close to 850 groups of conversations left.

Also please make sure that in the final dataset "sample_amazon_dataset.csv" you should only have these columns "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id,root_tweet_id,root_author_id,root_created_at,conversation_group_id
" and no other columns, and also they also make sure each group is considered atomic where either whole group conversation rows are selected or discarded.

You don't have to keep those extra columns "tweeth_count" or rule number column to be present or part of finally sampled dataset.

This ensure the good mixure of dataset for RAG.

## Expected output:
Your target is to write a python code to filter it out, and strictly make sure that all these rules are followed.
This python code is should be written preciously by keeping in mind the all thing I have stated above.