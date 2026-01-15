##==== Logistic Regression with Adult Dataset ====##
# Predict if an individual will earn more than $50K using logistic regression

# Load data from UCI repository
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
inputData <- read.csv(url, header = FALSE, strip.white = TRUE)

# Add column names
colnames(inputData) <- c("AGE", "WORKCLASS", "FNLWGT", "EDUCATION", "EDUCATIONNUM",
                         "MARITALSTATUS", "OCCUPATION", "RELATIONSHIP", "RACE", "SEX",
                         "CAPITALGAIN", "CAPITALLOSS", "HOURSPERWEEK", "NATIVECOUNTRY", "INCOME")

# Convert target to binary (1 = >50K, 0 = <=50K)
inputData$ABOVE50K <- ifelse(inputData$INCOME == ">50K", 1, 0)
inputData$INCOME <- NULL  # Remove original column

# Check structure and summary
head(inputData)
str(inputData)
table(inputData$ABOVE50K)
# Class bias exists: ~75% earn <=50K, ~25% earn >50K

## Create Balanced Training/Test Sets
input_ones <- inputData[which(inputData$ABOVE50K == 1), ]
input_zeros <- inputData[which(inputData$ABOVE50K == 0), ]

set.seed(0623)
input_ones_training_rows <- sample(1:nrow(input_ones), 0.7*nrow(input_ones))
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.7*nrow(input_ones))  # Match size of 1's

training_ones <- input_ones[input_ones_training_rows, ]  
training_zeros <- input_zeros[input_zeros_training_rows, ]
trainingData <- rbind(training_ones, training_zeros)  # Balanced training set

test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
testData <- rbind(test_ones, test_zeros)  # Test set

## Build Logistic Regression Model
logitMod <- glm(ABOVE50K ~ AGE + EDUCATIONNUM + HOURSPERWEEK + CAPITALGAIN + CAPITALLOSS, 
                data = trainingData, family = binomial(link = "logit"))

summary(logitMod)

## Model Coefficients (Odds Ratios)
exp(coef(logitMod))
# Each unit increase in AGE/EDUCATION/HOURS increases odds of earning >50K

## Predictions on Test Set
predicted <- predict(logitMod, testData, type = "response")  # Probabilities
predicted_class <- ifelse(predicted > 0.5, 1, 0)  # Binary classification

## Confusion Matrix
table(Predicted = predicted_class, Actual = testData$ABOVE50K)

## Accuracy
accuracy <- mean(predicted_class == testData$ABOVE50K)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

## ROC Curve and AUC
require(ROCR)
pr <- prediction(predicted, testData$ABOVE50K)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf, main = "ROC Curve - Adult Income Prediction")

# Calculate AUC
auc <- performance(pr, measure = "auc")
auc_value <- auc@y.values[[1]]
print(paste("AUC:", round(auc_value, 3)))

# AUC Interpretation:
# 0.9-1.0 = Excellent
# 0.8-0.9 = Good
# 0.7-0.8 = Fair
# 0.6-0.7 = Poor
# 0.5-0.6 = Fail

## Find Optimal Cutoff
acc_perf <- performance(pr, measure = "acc")
plot(acc_perf, main = "Accuracy vs Cutoff")

ind <- which.max(acc_perf@y.values[[1]])
best_acc <- acc_perf@y.values[[1]][ind]
best_cutoff <- acc_perf@x.values[[1]][ind]

c(Best_Accuracy = best_acc, Optimal_Cutoff = best_cutoff)

## Apply Optimal Cutoff
predicted_optimal <- ifelse(predicted > best_cutoff, 1, 0)
optimal_accuracy <- mean(predicted_optimal == testData$ABOVE50K)
print(paste("Optimized Accuracy:", round(optimal_accuracy * 100, 2), "%"))

## Sensitivity and Specificity at Optimal Cutoff
conf_matrix <- table(Predicted = predicted_optimal, Actual = testData$ABOVE50K)
sensitivity <- conf_matrix[2,2] / sum(conf_matrix[,2])  # True Positive Rate
specificity <- conf_matrix[1,1] / sum(conf_matrix[,1])  # True Negative Rate

print(paste("Sensitivity:", round(sensitivity, 3)))
print(paste("Specificity:", round(specificity, 3)))