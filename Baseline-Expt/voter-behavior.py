#Considering voter:
#Stochastic gradient descent and gradient boosting is run in sequence. The gradient boosting has higher accuracy. Both the accuracies are below (in percentage error):

#SGD: [0.063, 0.464, 4.141, 25.403, 18.947, 7.817, 13.331, 4.895, 0.898, 1.089]

#Gradient Boosting: [0.066, 0.42, 1.082, 0.0, 0.696, 2.683, 6.927, 4.781, 0.755, 0.858]

#The case where gradient boosting shows incorrect result but the SGD shows correct are counted. The result is below:

#6 out of 27170 records for attack-type: 1
#18 out of 27170 records for attack-type: 2
#37 out of 27170 records for attack-type: 3
#0 out of 27170 records for attack-type: 4
#37 out of 27170 records for attack-type: 5
#141 out of 27170 records for attack-type: 6
#495 out of 27170 records for attack-type: 7
#47 out of 27170 records for attack-type: 8
#24 out of 27170 records for attack-type: 9
#13 out of 27170 records for attack-type: 10


#Tested for the SGD and Naïve Bayes:
#SGD: [0.063, 0.464, 4.141, 25.403, 18.947, 7.817, 13.331, 4.895, 0.898, 1.089]

#Naïve Bayes: [0.905, 78.874, 73.633, 27.667, 26.165, 72.529, 29.86, 8.024, 50.213, 47.365]

#The case where SGD shows incorrect result but the Naïve Bayes shows correct are counted. The result is below:

#5 out of 27170 records for attack-type: 1
#29 out of 27170 records for attack-type: 2
#868 out of 27170 records for attack-type: 3
#7794 out of 27170 records for attack-type: 4
#6729 out of 27170 records for attack-type: 5
#1535 out of 27170 records for attack-type: 6
#5130 out of 27170 records for attack-type: 7
#108 out of 27170 records for attack-type: 8
#2 out of 27170 records for attack-type: 9
#8 out of 27170 records for attack-type: 10

#code:

Y_validation = [None]*100
Y_targetpred_compared = []
for j in range(0,len(y_test)):
        Y_targetpred_compared.append(1 if y_test[j] == target_pred[j] else 0)
    Y_validation[i] = Y_targetpred_compared

naivebayes_Exploits=Y_validation
#stochasticGradientDescent_Exploits=Y_validation

for j in range(1,11):
    count = 0;
    for i in range(0,len(gradientboosting_Exploits[j])):
        if (stochasticGradientDescent_Exploits[j][i] == 0 and gradientboosting_Exploits[j][i] ==1):
            count = count+1
        else:
            pass
    print count, "out of",len(gradientboosting_Exploits[j]), "records for attack-type:",j

#Doesn’t get appreciable numbers to look over it. So, it’s better to focus on improving accuracy rather than thinking about the use of such special data points.
