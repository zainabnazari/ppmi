
rm(list = ls())  # empty memory
options("warn"=0)  #print max 10 warnings on screen


library(caret)
library(party)
library(e1071)
library(Amelia)
library(reshape2)
library(rpart)
library(factoextra)

library(robustbase)
require(ROCR)
library(ggplot2)
library(randomForest)
library(kernlab)

library(klaR)
library(svmpath)
library(lessR)	
library(rpart.plot)

library(ipred)

library(adabag)   ####library for boosting
library(ada)
library(ROCR)
library(pROC)
library(sandwich)
work_path <- "/Users/zainabnazari/Desktop/aaaa1ltimate_files_mhpc_thesis/top_148_2_gene_ML_models"


filename_input_table="Log_TMM_top_148.txt"



####################################################################################
### FUNCTION to split data matrix into train and test set
## data matrix must be rows (patients) X colums (variables)
## partition. returns the two sub-matrices
####################################################################################

split_data_train_test = function(in_data, perc_train=0.8, in_seed=123) 
	{
	set.seed(in_seed)
	Nrows_train=floor(nrow(in_data)*perc_train)
	Nrows_test=nrow(in_data)-Nrows_train
	rows_train=sample(c(1:nrow(in_data)),size=Nrows_train,replace=FALSE)
	rows_test = setdiff(c(1:nrow(in_data)),rows_train)
	train_set = in_data[rows_train,]
	test_set = in_data[rows_test,]
	a=list(train_set=train_set, test_set=test_set,train_index=rows_train,test_index=rows_test)
	return(a)
	}


####################################################################################
################################### LOAD  DATA #####################################

setwd(work_path)

mydata<-read.table(file=filename_input_table, sep = "\t", quote = "\"",row.names=1,header=TRUE, fill=TRUE)  # re-read data into a dataframe with just numbers as real data


ID_patients=rownames(mydata)
name_variables=colnames(mydata)[-which(colnames(mydata)=="Class")]	# eliminate "Class" variable from the predictors, it is the dependent binary response !!




################################# MACHINE LEARNING #################################
#################################### 2024 March ####################################
####################################################################################
##################################### ADABOOST ##################################




n_rep=101 ### number of trials, make it ODD so the median is calculated!
perc_train_set=0.7
num_classifiers=400	## num trees in each model
#num_x_points_smoothed_ROC_curve=50
model_list=NULL
max_AUC=0		# reset
max_AUC_index=0	# reset

train_test_factor_matrix=array(data=NA,dim=c(nrow(mydata),n_rep)) ## matrix with columns with vectors of elements "train"/"test" for each patient, one column per trial
colnames(train_test_factor_matrix)=paste("trial_",c(1:n_rep),sep="")
rownames(train_test_factor_matrix)=rownames(mydata)

variable_importance=array(data=NA,dim=c(length(name_variables),n_rep))	## varImp for all trials
colnames(variable_importance)=paste("trial_",c(1:n_rep),sep="")
rownames(variable_importance)=name_variables

variable_importance_norm_1_0=array(data=NA,dim=c(length(name_variables),n_rep)) ## varImp for all trial, normalised to the max of each trial, so that to have scores from 1.0 (top variable) .... to 0.0
colnames(variable_importance_norm_1_0)=paste("trial_",c(1:n_rep),sep="")
rownames(variable_importance_norm_1_0)=name_variables

variable_importance_count_vars_model=array(data=NA,dim=c(length(name_variables),n_rep)) ## counts how many times a variable appears across thr trees of each model (trial)
colnames(variable_importance_count_vars_model)=paste("trial_",c(1:n_rep),sep="")
rownames(variable_importance_count_vars_model)=name_variables

out_parameters=array(data=NA,dim=c(6,n_rep))
colnames(out_parameters)=paste("trial_",c(1:n_rep),sep="")
rownames(out_parameters)=c("AUC","Accuracy","Sensitivity","Specificity","Kappa","Mcnemar_test")



grDevices::windows(width=12, height=10)
out_plot_name=paste("out_AdaBoost_ROC_Ntrials_",n_rep,"_median_AUC_red.png",sep="")
png(out_plot_name, width = 12, height = 10, units = 'in', res = 300)		# to plot at highres, needs dev.off() at the end of plotting

for(i in 1:n_rep)
	{ 

					print(c("Adaboost Interation num. ", i))
	
   mydata_original=mydata
   
   mydata$Class=factor( mydata$Class)
	
	curr_set=split_data_train_test(in_data=mydata,perc_train=perc_train_set,in_seed=as.numeric(Sys.time()))
	TrainSet = curr_set$train_set
	TestSet = curr_set$test_set
	train_test_factor_matrix[curr_set$train_index,i]="train"
	train_test_factor_matrix[curr_set$test_index,i]="test"


	
	adaboost_model = boosting(Class~., data=TrainSet, boos=TRUE,   mfinal=num_classifiers, control=rpart.control(cp=0.0001))			### new function for boosting make model

			
	model_list[[i]]=adaboost_model
	adaboost_pred <- predict(adaboost_model, newdata=TestSet)
	adaboost_mc <- confusionMatrix(as.factor(adaboost_pred$class), as.factor(TestSet$Class))

	out_parameters["Accuracy",i]=adaboost_mc$overall["Accuracy"]
	out_parameters["Kappa",i]=adaboost_mc$overall["Kappa"]
	out_parameters["Mcnemar_test",i]=adaboost_mc$overall["McnemarPValue"]
	out_parameters["Sensitivity",i]=adaboost_mc$byClass["Sensitivity"]
	out_parameters["Specificity",i]=adaboost_mc$byClass["Specificity"]


		### ROC curve , AUC
	 adaboost.prob.rocr <- prediction(adaboost_pred$prob[,2], TestSet$Class)
	 adaboost.ROC <- performance(adaboost.prob.rocr, "tpr","fpr")
	 adaboost.auc <- performance(adaboost.prob.rocr,"auc")
	 adaboost.auc <- unlist(slot(adaboost.auc, "y.values")) # Get the area under the curve

	 out_parameters["AUC",i]=adaboost.auc
				print(filename_input_table)
				 print(c("rep n. ", i, "  AUC= ", adaboost.auc,"Mean_AUC=",mean(out_parameters["AUC",],na.rm = TRUE),"MAX_AUC=",max(out_parameters["AUC",],na.rm = TRUE)) )
				print("     ")

	 if (i==1){
		plot(adaboost.ROC, col="grey", main="AdaBoost models ROC curves")
		} else {
		plot(adaboost.ROC, col="grey",add=TRUE)
		}

				### smooth ROC curve 

x=unlist(adaboost.ROC@x.values)
y=unlist(adaboost.ROC@y.values)
	if (i==1){
		plot(x,y ,col="grey", type="l",main="AdaBoost models ROC curves",xlim=c(0.0,1.0),ylim=c(0.0,1.0), xlab="False Positive Rate", ylab="True Positive Rate")
		} else {
		points(x,y, col="grey",type="l")
		}
		


 }

	## compose variable importance matrix across all trials
	## if I print adaboost_model$trees I get as output all the trees of the ensemble model
	## For each trial I extract the variable importance for all single classifiers (trees) of each ensemble model, than compose them for each the trial
	## by averaging for each variable. For each tree there are maby variables with no VarImp --> they have zero VarImp
	## this acceess varImp for single tree	adaboost_model$trees[[i=1...num_classifiers]]$variable.importance
	## or on the list of models  model_list[[tt=1...nrep]]$trees[[i=1...num_classifiers]]$variable.importance
	## also rescale varImp to 1.0...0.0 range and further compute a third version of varImp by counting how many times each
	## variable appears in each trial across all trees, by summing the "TRUE" in a boolean matrix with TRUE, for each tree, for variables included in the tree
	
					## matrix of varImp for all the trees of each trial will be summarized by average
	curr_trial_varImp_matrix=array(data=0, dim=c(length(name_variables),num_classifiers))
	colnames(curr_trial_varImp_matrix)=paste("tree_",c(1:num_classifiers),sep="")
	rownames(curr_trial_varImp_matrix)=name_variables
	
	curr_trial_varImp_norm_1_0_matrix=array(data=0, dim=c(length(name_variables),num_classifiers))  #  the varImp rescaled in 1...0 =max..min scale
	colnames(curr_trial_varImp_norm_1_0_matrix)=paste("tree_",c(1:num_classifiers),sep="")
	rownames(curr_trial_varImp_norm_1_0_matrix)=name_variables
	
	curr_trial_counts_model_vars_matrix=array(data=FALSE, dim=c(length(name_variables),num_classifiers))  #  TRUE where variable has varImp>0
	colnames(curr_trial_counts_model_vars_matrix)=paste("tree_",c(1:num_classifiers),sep="")
	rownames(curr_trial_counts_model_vars_matrix)=name_variables
	
for (curr_trial in 1:n_rep)		## compose variable importance matrix across all trials
	{
	curr_trial_varImp_matrix[,]=0		# reset
	curr_trial_varImp_norm_1_0_matrix[,]=0	# reset
	curr_trial_counts_model_vars_matrix[,]=FALSE	#reset
	for (curr_tree in 1:num_classifiers)
		{
		curr_tree_struct=model_list[[curr_trial]]$trees[[curr_tree]]
		curr_tree_vars=setdiff(unique(curr_tree_struct$frame$var),"<leaf>")
	
		curr_tree_varImp=curr_tree_struct$variable.importance		### access varImp of each tree for each ensemble model (trial)
		non_zero_varImp_index=match(names(curr_tree_varImp), name_variables	,nomatch=FALSE)	# find the index position in variables vector of non zero varImp
		curr_trial_varImp_matrix[non_zero_varImp_index,curr_tree]=curr_tree_varImp
		curr_trial_varImp_norm_1_0_matrix[non_zero_varImp_index,curr_tree]=curr_tree_varImp/max(curr_tree_varImp)
		
		curr_trial_counts_model_vars_matrix[,curr_tree]=match(name_variables,curr_tree_vars,nomatch=FALSE)>0	# inverse set=TRUE  variable position whose varImp>0
			
		}	
	variable_importance[,curr_trial]=rowMeans(curr_trial_varImp_matrix)	### average across trees for each trial
	variable_importance_norm_1_0[,curr_trial]=rowMeans(curr_trial_varImp_norm_1_0_matrix)/max(rowMeans(curr_trial_varImp_norm_1_0_matrix))	### average across trees for each trial and rescale to 1.0 ...0.0
	variable_importance_count_vars_model[,curr_trial]=rowSums(curr_trial_counts_model_vars_matrix)  ## sum occurnmces of each variable in tress
	}

median_AUC_index=which(out_parameters["AUC",]==median(out_parameters["AUC",]))[1]
median_AdaBoost_model=model_list[[median_AUC_index]]
save(median_AdaBoost_model, file="out_AdaBoost_median_model.RData")	### write save best model	

			# recompute the AUC of median model among trial, using the corresponding test set
median_TestSet=mydata[which(train_test_factor_matrix[,median_AUC_index]=="test"),]
adaboost_pred <- predict(median_AdaBoost_model, newdata=median_TestSet)
adaboost.prob.rocr <- prediction(adaboost_pred$prob[,2], median_TestSet$Class)
adaboost.ROC <- performance(adaboost.prob.rocr, "tpr","fpr")
x=unlist(adaboost.ROC@x.values)
y=unlist(adaboost.ROC@y.values)	
points(x,y,col="purple",type="l",lwd=2)

dev.off() 


max_AUC_index=which(out_parameters["AUC",]==max(out_parameters["AUC",]))[1]
best_AdaBoost_model=model_list[[max_AUC_index]]
save(best_AdaBoost_model, file="out_AdaBoost_best_model.RData")	### write save best model	

			# recompute the AUC of best model among trial, using the corresponding test set
best_TestSet=mydata[which(train_test_factor_matrix[,max_AUC_index]=="test"),]
adaboost_pred <- predict(best_AdaBoost_model, newdata=best_TestSet)
adaboost.prob.rocr <- prediction(adaboost_pred$prob[,2], best_TestSet$Class)
adaboost.ROC <- performance(adaboost.prob.rocr, "tpr","fpr")
adaboost.auc <- performance(adaboost.prob.rocr,"auc")
best_AUC <- unlist(slot(adaboost.auc, "y.values")) # Get the area under the curve

grDevices::windows(width=12, height=10)
out_plot_name=paste("out_AdaBoost_Best_model_ROC.png",sep="")
png(out_plot_name, width = 12, height = 10, units = 'in', res = 300)		# to plot at highres, needs dev.off() at the end of plotting

plot(adaboost.ROC, col="purple", main=paste("AdaBoost best model, AUC=",best_AUC,sep=""))
dev.off()






write.table(variable_importance, "out_AdaBoost_varImp.txt", sep="\t",row.names=TRUE, col.names=TRUE)		## AUC sensitivity .... other parameters  for each trial
write.table(variable_importance_norm_1_0, "out_AdaBoost_varImp_norm_1_0.txt", sep="\t",row.names=TRUE, col.names=TRUE)		## AUC sensitivity .... other parameters  for each trial
write.table(variable_importance_count_vars_model, "out_AdaBoost_counts_vars_models.txt", sep="\t",row.names=TRUE, col.names=TRUE)		## AUC sensitivity .... other parameters  for each trial

write.table(out_parameters, "out_AdaBoost_parameters.txt", sep="\t",row.names=TRUE, col.names=TRUE)		## AUC sensitivity .... other parameters  for each trial
write.table(train_test_factor_matrix, "out_AdaBoost_train_test_split_matrix.txt" ,sep="\t",row.names=TRUE, col.names=TRUE)  # for each trial tein/test patients splitting



#####################  End ADABOOST
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################



