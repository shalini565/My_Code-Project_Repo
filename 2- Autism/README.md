### Machine Learning Approaches for the Classification Problem for Autism Spectrum Disorder 

#### Flow of the project

![image](https://user-images.githubusercontent.com/65540050/164676878-e8c4b824-3fc6-4819-ac5a-68f7b71d20a0.png)

 

#### ABOUT THE DATASET:

Here I am using a dataset available on Kaggle related to autism screening of adults that contained 20 features to be utilized for further analysis especially in determining influential autistic traits and improving the classification of ASD cases.

#### This data contains 704 instances, and contains the following attributes:
*	Age: number (Age in years).
*	Gender: String [Male/Female].
*	Ethnicity: String (List of common ethnicities in text format).
* Jaundice: Boolean [yes or no].
*	Family member with PDD: Boolean [yes or no].
*	Who is completing the test: String [Parent, self, caregiver, medical staff, clinician etc.].
*	Country of residence: String (List of countries in text format).
*	Used the screening app before : Boolean [yes or no] (Whether the user has used a screening app)
*	Screening Method Type: Integer [0, 1, 2, 3] (The type of screening methods chosen based on age category (0=toddler, 1=child, 2= adolescent, 3= adult).
*	Question 1-10 Answer: Binary [0, 1] (The answer code of the question based on the screening method used).
*	Screening Score: Integer (The final score obtained based on the scoring algorithm of the screening method used. This was computed in an automated manner).

#### RESULTS:

Selection of classification algorithms can be done based on evaluation metrics such as accuracy, f-score, Recall, Precision and AUC.
K-Nearest-Neighbours (KNN):  K is an example of a hyper parameter - a parameter on the model itself which can be tuned for best results on our dataset.
For KNN, Choosing K is tricky, so can't discard KNN until I've tried different values of K. Hence I wrote a for loop to run KNN with K values ranging from 10 to 50 and see if K makes a substantial difference.

