# Animal Shelter Cat Adoption Likelihood ML Project
	
## Abstract
This supervised machine learning project aims to predict animal shelter adoption likelihood for cats using the Animal Shelter Intakes and Outcomes dataset from Long Beach. The dataset includes information on primary colors, sex, intake conditions, intake types, reasons for intake, outcome types, outcome subtypes, and indicators for intake and outcome mortality. The data will be utilized to train a classification model that determines whether an animal is likely to be adopted or not. Model performance will be evaluated through metrics such as ROC-AUC and F1 score, using cross-validation and testing on unseen data.

the highlighted parts will change depending on data and proposed solution.
Background (Ezra)
Free-ranging cats can cause serious damage to the ecosystem as they have been introduced to the ecosystem by humans. Cats are natural predators that have greatly contributed to the extinction of 33 wildlife species worldwide [1], which is why it is the responsibility of humans to help keep cats off the streets in a humane way. Animal shelters play an important role in rescuing these homeless, free-ranging cats. Every year, around 6.5 million animals enter animal shelters nationwide. Due to lack of resources and funding, it is very challenging to find permanent homes for these cats. Because of this, of the 6.5 million shelter animals that are rescued, approximately 920,000 of them are euthanized with the majority of these euthanasias being performed on cats [2]. To prevent the further unnecessary deaths of cats, we will analyze what features make cats appealing to adopt in animal shelters in order to help animal shelters increase cat adoptions in the U.S.

Although there is no prior work done on the animal shelter outcomes of specifically shelter cats, there have been classification models done to predict the outcome of shelter animals. Danielle’s work aimed to predict an animal’s outcome from the shelter based on their breed, color, sex, and age using random forests classifier to predict the outcome of the animal. Although her model had an f-1 score of 0.73, her model would incorrectly classify “adoption” as “return to owner” and would misclassify the outcomes of popular breeds as “transfer” or “death” when the true value was “euthanasia” [3]. In Mohammad’s work, it also aims to predict the outcomes of animals in animal shelters by using a decision tree classifier [4]. For our project, instead of predicting the outcomes of animals, we will predict the likelihood of a cat being adopted based on their attributes.

[1] https://www.fws.gov/sites/default/files/documents/keep-cats-indoors.pdf
[2] https://www.aspca.org/helping-people-pets/shelter-intake-and-surrender/pet-statistics
[3] https://github.com/danielle-altshuler/shelter_animal_predictions
[3] https://github.com/mushahidq/animal_shelter_outcome_prediction/tree/main

## Problem Statement
The goal of this project is to forecast the possibility of cat adoption from shelters. Using parameters including primary color, sex, intake conditions, intake kinds, reasons for intake, outcome types, outcome subtypes, and indications for intake and outcome mortality, the goal is to create a model that can reliably categorize cats as either likely to be adopted or not.
We can develop a predictive model that can help shelter personnel identify animals with a higher likelihood of being adopted by training it on historical data from animal shelter intakes and outcomes. This approach can help animal shelters operate more successfully and efficiently overall by streamlining the adoption process and better allocating resources.
The issue can be quantified because it entails foretelling a binary outcome (adoption likelihood) based on quantitative characteristics. Mathematically, it can be described as a binary classification task in which the model discovers relationships and patterns in the data to produce predictions.
The issue is measurable because it can be assessed using several performance criteria like ROC-AUC and the F1 score. These measures will help us determine how accurately the model can categorize animals as being likely to be adopted or not.
Because it can be duplicated and happens frequently in animal shelter settings, the issue is reproducible. Over time, new data can be gathered, allowing the model to be continually updated and trained to increase its prediction power.

## Data

For our project, we will be employing a subset of the City of Long Beach's Animal Shelter data. This dataset encompasses a broad array of animal types, including dogs, birds, and wild animals. However, our analysis will be limited to cats, which constitute approximately 50 percent of the total data. We've made this decision due to the sparseness of data points related to other animals in contrast to the number of variables. This disparity would make it challenging to generate accurate or substantial predictions for animal categories outside of "Cats".

* [hyperlink to Dataset](https://data.longbeach.gov/explore/dataset/animal-shelter-intakes-and-outcomes/table/?disjunctive.animal_type&disjunctive.primary_color&disjunctive.sex&disjunctive.intake_cond&disjunctive.intake_type&disjunctive.reason&disjunctive.outcome_type&disjunctive.outcome_subtype&disjunctive.intake_is_dead&disjunctive.outcome_is_dead&dataChart=eyJxdWVyaWVzIjpbeyJjaGFydHMiOlt7ImFsaWduTW9udGgiOnRydWUsInR5cGUiOiJjb2x1bW4iLCJmdW5jIjoiQ09VTlQiLCJ5QXhpcyI6ImFnZSIsInNjaWVudGlmaWNEaXNwbGF5Ijp0cnVlLCJjb2xvciI6InJhbmdlLWN1c3RvbSIsInBvc2l0aW9uIjoiY2VudGVyIn1dLCJ4QXhpcyI6ImludGFrZV9kYXRlIiwibWF4cG9pbnRzIjpudWxsLCJ0aW1lc2NhbGUiOiJtb250aCIsInNvcnQiOiIiLCJzZXJpZXNCcmVha2Rvd25UaW1lc2NhbGUiOiIiLCJzZXJpZXNCcmVha2Rvd24iOiJpbnRha2VfaXNfZGVhZCIsInN0YWNrZWQiOiJub3JtYWwiLCJjb25maWciOnsiZGF0YXNldCI6ImFuaW1hbC1zaGVsdGVyLWludGFrZXMtYW5kLW91dGNvbWVzIiwib3B0aW9ucyI6eyJkaXNqdW5jdGl2ZS5hbmltYWxfdHlwZSI6dHJ1ZSwiZGlzanVuY3RpdmUucHJpbWFyeV9jb2xvciI6dHJ1ZSwiZGlzanVuY3RpdmUuc2V4Ijp0cnVlLCJkaXNqdW5jdGl2ZS5pbnRha2VfY29uZCI6dHJ1ZSwiZGlzanVuY3RpdmUuaW50YWtlX3R5cGUiOnRydWUsImRpc2p1bmN0aXZlLnJlYXNvbiI6dHJ1ZSwiZGlzanVuY3RpdmUub3V0Y29tZV90eXBlIjp0cnVlLCJkaXNqdW5jdGl2ZS5vdXRjb21lX3N1YnR5cGUiOnRydWUsImRpc2p1bmN0aXZlLmludGFrZV9pc19kZWFkIjp0cnVlLCJkaXNqdW5jdGl2ZS5vdXRjb21lX2lzX2RlYWQiOnRydWV9fX1dLCJkaXNwbGF5TGVnZW5kIjp0cnVlLCJhbGlnbk1vbnRoIjp0cnVlLCJ0aW1lc2NhbGUiOiIifQ%3D%3D)
* Size of dataset: 17,292 observations, 15 variables
* Critical Variables: Age, Intake type, outcome type and Reason for intake. These variables will be one hot encoded.



## Proposed Solution

The primary function of our model will be to use various features (such as primary color, sex, intake conditions, intake types, reasons for intake, outcome types, outcome subtypes, and indications for intake and outcome mortality) to __predict a binary outcome__: whether a cat is likely to have a good outcome (adoption, return to owner etc) or a bad outcome (euthanasia, transfer etc). The outcomes have various different categories, but can be simplified into a good or bad binary.

As for a benchmark model, we will start with a simple __logistic regression__ model given its interpretability and suitability for binary classification tasks. This model will provide a baseline against which we can compare the performance of more complex models. If other models do not significantly outperform the logistic regression model, its simplicity and interpretability might make it the preferred choice.

To ensure the model's applicability, we'll preprocess the data by one-hot encoding categorical variables like age, intake type, outcome type, and reason for intake. This conversion of categorical data into a format that can be provided to machine learning algorithms will ensure our model can utilize all available information.

The implementation of the solution will be done in Python, using libraries like pandas for data manipulation, sklearn for machine learning algorithms, and matplotlib or seaborn for visualizations. Each step will be clearly documented in a Jupyter notebook or google colab.


## Evaluation Metrics

The primary evaluation metric we intend to use is the Receiver Operating Characteristic for Area Under the Curve (ROC-AUC, or AUC for short) metric. We decided to go with this metric because it works well with the algorithm that we have decided to use (logistic regression). AUC is appropriate for logit because it allows us to determine how well the model correctly predicts classes. While our project's issue is essential, it isn't as dire as other machine learning projects like cancer testing. Indicating that a cat won't get adopted when it actually will is a pleasant surprise instead of a life-or-death situation. Therefore, we have more leeway regarding classification levels and don't have to worry about the severity of picking something like a false negative over a false positive. ROC AUC compares the false classification rates at all classification thresholds, making it ideal for a model that isn't making high-stakes predictions. It also allows us to change the classification rate without significant concern for the consequences of false predictions. The ROC AUC curve compares the true positive rate (recall) to the false positive rate (1-specificity). We calculate these as follows:

__Recall = true positives/(true positives + false negatives)__

__Specificity = true negatives/(true negatives + false positives)__

The greater the area under the curve (AUC), the better the model distinguishes between the two classes. The best possible "AOC" would be a value of 1, indicating that the model can perfectly differentiate the two classes. The worst possible "AOC" we can have is 0.5, meaning that the model cannot distinguish the two classes and therefore predicts by chance. To plot this curve, we use different classification thresholds and find the TPR and FPR based on those thresholds. We then plot the TPR and FPR on the x and y axes. 

We also plan to use the F-score metric. The F-score is appropriate for this model because it is another way of indicating its performance concerning the bias-variance tradeoff. The F-score shows a weighted average of the precision and recall scores, giving us a metric that favors a balance between the two. The F-score will be lower if the precision or recall is higher than the other metric. We calculate precision using the following formula:

__Precision = true positives/(true positives + false positives)__

The F score is calculated by using the following formula:

__F-score = 2 * (precision * recall) / (precision + recall)__

## Ethics & Privacy

In terms of ethics and privacy concerns, there should not be many obvious ethical implications since the data collected is not on human subjects. However, there is an issue of privacy with the data as the Long Beach animal shelter has a record of where the cats were taken from. This may be a privacy issue because the addresses of owners who have chosen to surrender their cat are published in the dataset and it is unknown to us if this data was collected with the owners’ permission. This is especially a concern if the cat was taken due to mistreatment and/or abuse as animal abusers can possibly be tracked down and harassed with this data. However, we do not plan on using any locational data in our model so it should not be an issue. 

## Team Expectations

* Team members will use discord as a primary means of communication
* Team members are aware of each other’s schedules through a when2meet form
* Team members are expected to be punctual to meetings and communicate their absences if unforeseen situations arise
* Team Members are expected to read and follow the prescribed deadlines in the Project Timeline Proposal
* Team Members should feel comfortable sharing their issues and expectations of other members 


## Project Timeline Proposal


| __Date__    | __Time__ | __Completed before meeting__    | __Discuss at meeting__ |
| -------- | ------- |------- |------- |
| 5/15 | 8PM | Finding ideas, datasets for the project (All) | Which subject we’re going with, dividing the tasks for the proposal,  draft project proposal |
| 5/16 | 5PM | Divide up tasks for Project proposal and complete parts separately | Review/Edit/Talk about our parts and what we need to each change |
| 5/17 | Deadline Project Proposal |
| 5/19 | 3PM | Import & Wrangle Data (Undetermined) | Discuss Analysis Plan, split tasks |
| 5/23 | 5PM | Start project code according to tasks (All) | Discuss progress & changes if needed |
| 5/26 | 3PM | Read the peer reviews and change according to previous Project Proposal tasks (Everyone) | Discuss progress & changes if needed |
| 5/30 | 5PM | Finish tasks (Everyone) | Finalizing details before deadline |
| 5/31 | Deadline Checkpoint |
| 6/6 | 5PM | Complete analysis | Draft results/conclusion/discussion, split tasks |
| 6/13 | 5PM | Study results  | Complete project |
| 6/14 | Deadline Final Project |

