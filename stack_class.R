## PACOTES ##

library(tidyverse)	# manipulacao de dados
library(tidymodels) # ferramentas de ML
library(stacks)     # stacking
library(cutpointr)  # ponto de corte
library(tictoc)     # registrar o tempo de execução de comandos



## DADOS ##

df<- read.csv("winequality_red.csv") %>% 
  data.frame()

glimpse(df)

df<- df %>% mutate(quality=ifelse(quality>=7,"good","bad"))
df$quality<- df$quality %>% as.factor()
glimpse(df)




##### SPLIT TRAIN/TEST/VALIDATION #####

set.seed(0)
#split<- initial_split(df, strata=quality)
split<- initial_validation_split(df, prop=c(0.6,0.2), strata=quality)


df.train<- training(split)  # usado para cross-validation
df.valid<- validation(split)  # usado para determinar ponto de corte
df.test<- testing(split)    # usado para verificar desempenho


folds<- vfold_cv(df.train, v=3, strata=quality)




##### PRÉ-PROCESSAMENTO #####

receita<- recipe(quality ~ . , data = df.train) %>%
  step_filter_missing(all_predictors(),threshold = 0.4) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors())




##### MODELOS #####

model_knn<- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

model_net<- logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

model_rfo<- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("classification")

model_svm<- svm_linear(cost = tune(), margin = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")



##### WORKFLOW #####

wf_knn<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_knn)

wf_net<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_net)

wf_rfo<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_rfo)

wf_svm<- workflow() %>%
  add_recipe(receita) %>%
  add_model(model_svm)



##### TUNAGEM DE HIPERPARAMETROS - BAYESIAN SEARCH #####

## KNN

tic()
tune_knn<- tune_bayes(wf_knn,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(neighbors(range=c(1,30)))
)
toc()
# 17.98 sec elapsed




## NET - ELASTIC NET

tic()
tune_net<- tune_bayes(wf_net,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(penalty(range=c(-10,10)),
                                              mixture(range=c(0,1)))
)
toc()
# 43.83 sec elapsed




## RFO - RANDOM FOREST

tic()
tune_rfo<- tune_bayes(wf_rfo,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(mtry(range=c(1,10)),
                                              trees(range=c(50,10000)),
                                              min_n(range=c(1,40)))
)
toc()
# 156.77 sec elapsed (~ 2.5 min)




## SVM - SUPPORT VECTOR MACHINE

tic()
tune_svm<- tune_bayes(wf_svm,
                      resamples = folds,
                      initial = 10,
                      #control = control_stack_bayes(),
                      control = control_bayes(save_pred=TRUE,
                                              save_workflow=TRUE,
                                              seed=0),
                      metrics = metric_set(roc_auc),
                      param_info = parameters(cost(range=c(-10,5)),
                                              svm_margin(range=c(0,0.5)))
)
toc()
# 42.54 sec elapsed




##### PREPARANDO STACKING #####

stack_ensemble_data<- stacks() %>% 
  add_candidates(tune_knn) %>% 
  add_candidates(tune_net) %>% 
  add_candidates(tune_rfo) %>% 
  add_candidates(tune_svm)

stack_ensemble_data





##### AJUSTANDO STACKING #####

set.seed(0)
stack_ensemble_model<- stack_ensemble_data %>% 
  blend_predictions(penalty = 10^(-9:1),
                    mixture = 0.5, # 0=RIDGE; 1=LASSO
                    control = control_grid(),
                    non_negative = TRUE,
                    metric = metric_set(roc_auc))

autoplot(stack_ensemble_model)
autoplot(stack_ensemble_model,type = "weights")

stack_ensemble_model




##### FINALIZANDO O MODELO #####

stack_ensemble_model<- stack_ensemble_model %>% 
  fit_members()

stack_ensemble_model




##### FINALIZANDO MODELOS INDIVIDUAIS #####

wf_train_knn<- wf_knn %>% finalize_workflow(select_best(tune_knn)) %>% fit(df.train)
wf_train_net<- wf_net %>% finalize_workflow(select_best(tune_net)) %>% fit(df.train)
wf_train_rfo<- wf_rfo %>% finalize_workflow(select_best(tune_rfo)) %>% fit(df.train)
wf_train_svm<- wf_svm %>% finalize_workflow(select_best(tune_svm)) %>% fit(df.train)




####################################################
#####  ESCOLHENDO O PONTO DE CORTE - F1-SCORE  #####
####################################################

# PROBABILIDADES (DADOS VALIDACAO)

prob.knn<- wf_train_knn %>% predict(df.valid, type="prob")
prob.net<- wf_train_net %>% predict(df.valid, type="prob")
prob.rfo<- wf_train_rfo %>% predict(df.valid, type="prob")
prob.svm<- wf_train_svm %>% predict(df.valid, type="prob")
prob.stc<- stack_ensemble_model %>% predict(df.valid, type="prob")

df.prob<- cbind.data.frame(df.valid$quality,
                           prob.knn[,2],
                           prob.net[,2],
                           prob.rfo[,2],
                           prob.svm[,2],
                           prob.stc[,2])

colnames(df.prob)<- c("quality",
                      "knn",
                      "net",
                      "rfo",
                      "svm",
                      "stc")

df.prob %>% head()    # VISUALIZANDO PROBABILIDADES


# CORTE

cut_knn<- cutpointr(df.prob, knn, quality, method=maximize_metric, metric=F1_score)$optimal_cutpoint
cut_net<- cutpointr(df.prob, net, quality, method=maximize_metric, metric=F1_score)$optimal_cutpoint
cut_rfo<- cutpointr(df.prob, rfo, quality, method=maximize_metric, metric=F1_score)$optimal_cutpoint
cut_svm<- cutpointr(df.prob, svm, quality, method=maximize_metric, metric=F1_score)$optimal_cutpoint
cut_stc<- cutpointr(df.prob, stc, quality, method=maximize_metric, metric=F1_score)$optimal_cutpoint

cbind(cut_knn,
      cut_net,
      cut_rfo,
      cut_svm,
      cut_stc)



# PREDIZENDO CLASSES (CLASSIFICACAO - DADOS TESTE)

prob.knn<- wf_train_knn %>% predict(df.test, type="prob")
prob.net<- wf_train_net %>% predict(df.test, type="prob")
prob.rfo<- wf_train_rfo %>% predict(df.test, type="prob")
prob.svm<- wf_train_svm %>% predict(df.test, type="prob")
prob.stc<- stack_ensemble_model %>% predict(df.test, type="prob")

df.prob<- cbind.data.frame(df.test$quality,
                           prob.knn[,2],
                           prob.net[,2],
                           prob.rfo[,2],
                           prob.svm[,2],
                           prob.stc[,2])

colnames(df.prob)<- c("quality",
                      "knn",
                      "net",
                      "rfo",
                      "svm",
                      "stc")

df.prob %>% head()    # VISUALIZANDO PROBABILIDADES

df.pred.class<- df.prob %>% 
  mutate(knn=ifelse(knn>cut_knn,"good","bad")) %>% 
  mutate(net=ifelse(net>cut_net,"good","bad")) %>% 
  mutate(rfo=ifelse(rfo>cut_rfo,"good","bad")) %>% 
  mutate(svm=ifelse(svm>cut_svm,"good","bad")) %>% 
  mutate(stc=ifelse(stc>cut_stc,"good","bad")) %>% 
  mutate(across(!quality, as.factor))

df.pred.class %>% head()    # VISUALIZANDO CLASSES




#####  VERIFICANDO MEDIDAS DE CLASSIFICAÇÃO  #####

medidas<- cbind(summary(conf_mat(df.pred.class, quality, knn))[,-2],
                summary(conf_mat(df.pred.class, quality, net))[,3],
                summary(conf_mat(df.pred.class, quality, rfo))[,3],
                summary(conf_mat(df.pred.class, quality, svm))[,3],
                summary(conf_mat(df.pred.class, quality, stc))[,3])                     

colnames(medidas)<- c("medida",
                      "knn",
                      "net",
                      "rfo",
                      "svm",
                      "stc")

medidas
