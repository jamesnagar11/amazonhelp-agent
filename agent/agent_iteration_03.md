### Iteration 3: (Goal: sampling 200 golden dataset from filtered_amazon_dataset.csv)

Please follow ./agent_iteration_02.md to understand the iteration 3 but with just only a single change.
The only single change is that instead of dividing dataset into 850 groups, I want you to divide the dataset into 200 groups and also keep in mind the edge cases from 1 to 17 and the rules which I have mentioned in the ./agent_iteration_02.md and then create a sample dataset named as "golden_dataset.csv" with the same columns and with just this new column added that represents for a single group of conversation that which rules does this group follows and this new column is named "rule_number"

And this time, from previous iteration 2 , i want you to sample list of groups for each ruleset number, but when you try to find percentage of each rule contribution in 100% among all rules, I want you to iterate over rule numbers in sorted order from rule number 1 to 17 and if rules are missing then those are obiviously skipped and you can use sorted set to iterate , and then I want you to from each ruleset randomly select a conversation_group_id and consider the conversation rows of this group_id and then move to next ruleset in sorted order, and again perform same operation of random conversation_group_id selection until you have selected 200 groups in total, and each conversation group id from the dataset that is selected is atomic, which means if a conversation_group_id is selected then all its consecutive rows should be included in the final dataset and if it is not selected then none of its consecutive rows should be included in the final dataset.
Do this until you have selected 200 groups, and make sure when you randomly chose then make sure that it's not already selected by you.

And in this manner finally output this sampled data in "golden_dataset.csv"

## Expected output:
Your target is to write a python code to filter it out, and strictly make sure that all these rules are followed.
This python code is should be written preciously by keeping in mind the all thing I have stated above.